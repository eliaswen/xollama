package llm

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"
)

// noticeEngine is an engine that declares generation_notice_v1 in /props
// (when declares is true) and answers POST /v1/notice for the keys given.
func noticeEngine(t *testing.T, declares bool, running map[string]bool) (*llamaServerRunner, *[]string) {
	t.Helper()
	var asked []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/props":
			f := []string{"polykv_v1"}
			if declares {
				f = append(f, featureGenerationNotice)
			}
			json.NewEncoder(w).Encode(map[string]any{"features": f})
		case "/v1/notice":
			b, _ := io.ReadAll(r.Body)
			asked = append(asked, string(b))
			var req struct{ Key, Kind string }
			json.Unmarshal(b, &req)
			if !running[req.Key] {
				w.WriteHeader(http.StatusNotFound)
				w.Write([]byte(`{"error":"no running request with that key"}`))
				return
			}
			w.Write([]byte(`{"ok":true}`))
		}
	}))
	t.Cleanup(srv.Close)
	u, _ := url.Parse(srv.URL)
	port, _ := strconv.Atoi(u.Port())
	return &llamaServerRunner{port: port, client: srv.Client(), usedOpencoti: true}, &asked
}

const clientNotices = `{"key":"sess:abc","think":[{"at_percent":60}],"answer":[{"at_percent":80}]}`

func TestNoticesGoOnlyToAnEngineThatPlantsThem(t *testing.T) {
	s, _ := noticeEngine(t, true, nil)
	if got := s.noticesFor(t.Context(), json.RawMessage(clientNotices)); string(got) != clientNotices {
		t.Fatalf("an engine declaring generation_notice_v1 got %q, want the client's object unchanged", got)
	}

	old, _ := noticeEngine(t, false, nil)
	if got := old.noticesFor(t.Context(), json.RawMessage(clientNotices)); got != nil {
		t.Fatalf("an engine without the feature got %q", got)
	}

	// Stock llama.cpp: the body stays upstream's, no notices key at all.
	stock := &llamaServerRunner{}
	if got := stock.noticesFor(t.Context(), json.RawMessage(clientNotices)); got != nil {
		t.Fatalf("stock llama.cpp got %q", got)
	}
	b, _ := json.Marshal(llamaServerCompletionRequest{Prompt: "hi"})
	if strings.Contains(string(b), "notices") {
		t.Fatalf("a request without notices carries the key: %s", b)
	}
}

func TestTheWaitingCallReachesTheRunnerHoldingTheKey(t *testing.T) {
	s, asked := noticeEngine(t, true, map[string]bool{"sess:abc": true})

	if status, _, err := PostNotice(t.Context(), "sess:abc", "message_waiting"); status != http.StatusNotFound || err != ErrNoNoticeKey {
		t.Fatalf("before the request runs: %d %v, want 404 and no engine call", status, err)
	}

	release := s.trackNotice(json.RawMessage(clientNotices))
	status, body, err := PostNotice(t.Context(), "sess:abc", "message_waiting")
	if err != nil || status != http.StatusOK || !strings.Contains(string(body), `"ok":true`) {
		t.Fatalf("while it runs: %d %s %v", status, body, err)
	}
	if len(*asked) != 1 || (*asked)[0] != `{"key":"sess:abc","kind":"message_waiting"}` {
		t.Fatalf("engine was asked %q", *asked)
	}

	release()
	if status, _, _ := PostNotice(t.Context(), "sess:abc", "message_waiting"); status != http.StatusNotFound {
		t.Fatalf("after the request ended: %d, want 404", status)
	}
}

func TestAKeyHeldByARunningRequestIsNotTakenOver(t *testing.T) {
	a, _ := noticeEngine(t, true, nil)
	b, _ := noticeEngine(t, true, nil)
	releaseA := a.trackNotice(json.RawMessage(clientNotices))
	defer releaseA()
	releaseB := b.trackNotice(json.RawMessage(clientNotices))
	releaseB()
	noticeKeys.Lock()
	held := noticeKeys.m["sess:abc"]
	noticeKeys.Unlock()
	if held != a {
		t.Fatal("the second request's release removed the first request's key")
	}
}
