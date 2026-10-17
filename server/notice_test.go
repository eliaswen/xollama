package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func withNoticesServed(t *testing.T, on bool) {
	t.Helper()
	old := noticesServed
	noticesServed = func() bool { return on }
	t.Cleanup(func() { noticesServed = old })
}

func identity(t *testing.T) api.XollamaIdentity {
	t.Helper()
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.GET(api.XollamaIdentityPath, XollamaIdentityHandler)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, api.XollamaIdentityPath, nil))
	var id api.XollamaIdentity
	if err := json.Unmarshal(w.Body.Bytes(), &id); err != nil {
		t.Fatal(err)
	}
	return id
}

// A client takes the spans it hides only from the block: without it the
// planted notices would show in its chat (ollama #1449), so the feature never
// goes out alone.
func TestNoticesAreAdvertisedWithTheirHideSpansOnlyWhenTheEngineHasThem(t *testing.T) {
	withNoticesServed(t, true)
	id := identity(t)
	if !slices.Contains(id.Features, api.FeatureGenerationNotice) {
		t.Fatalf("features %v lack %s", id.Features, api.FeatureGenerationNotice)
	}
	g := id.GenerationNotice
	if g == nil || g.Version != 2 || len(g.Hide) != 1 || g.Hide[0].Open != "<engine_status" || g.Hide[0].Close == nil || *g.Hide[0].Close != "</engine_status>" {
		t.Fatalf("generation_notice = %+v, want version 2 hiding <engine_status ... </engine_status>", g)
	}

	withNoticesServed(t, false)
	id = identity(t)
	if slices.Contains(id.Features, api.FeatureGenerationNotice) || id.GenerationNotice != nil {
		t.Fatalf("an engine without notices advertised them: %v %+v", id.Features, id.GenerationNotice)
	}
}

// The hook line in ChatHandler: the client's object reaches the runner
// unchanged.
func TestAChatRequestsNoticesReachTheRunner(t *testing.T) {
	mock := &mockRunner{}
	mock.CompletionFn = func(ctx context.Context, r llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
		fn(llm.CompletionResponse{Content: "hi"})
		fn(llm.CompletionResponse{Done: true, DoneReason: llm.DoneReasonStop})
		return nil
	}
	gin.SetMode(gin.TestMode)
	s := newServerWithMockRunner(t, mock)
	createMinimalGGUFModel(t, s, "nt", nil, "{{ .Prompt }}", nil)

	notices := json.RawMessage(`{"key":"s:1","answer":[{"at_percent":80}]}`)
	w := createRequest(t, s.ChatHandler, api.ChatRequest{Model: "nt", Messages: []api.Message{{Role: "user", Content: "x"}}, Notices: notices})
	if w.Code != http.StatusOK {
		t.Fatalf("status %d %s", w.Code, w.Body)
	}
	if string(mock.CompletionRequest.Notices) != string(notices) {
		t.Fatalf("runner got notices %q, want %q", mock.CompletionRequest.Notices, notices)
	}
}

func TestTheWaitingCallRefusesWhatContractV1DoesNotHave(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{}
	r := gin.New()
	r.POST(api.XollamaNoticePath, s.NoticeHandler)
	for body, want := range map[string]int{
		`{"key":"s:1","kind":"message_text"}`:     http.StatusBadRequest,
		`{"kind":"message_waiting"}`:              http.StatusBadRequest,
		`not json`:                                http.StatusBadRequest,
		`{"key":"gone","kind":"message_waiting"}`: http.StatusNotFound,
	} {
		w := httptest.NewRecorder()
		r.ServeHTTP(w, httptest.NewRequest(http.MethodPost, api.XollamaNoticePath, strings.NewReader(body)))
		if w.Code != want {
			t.Errorf("%s: status %d, want %d (%s)", body, w.Code, want, w.Body)
		}
	}
}

// context_window_unbooked_v1 is advertised exactly while num_ctx_hold reaches
// the engine.
func TestUnbookedWindowsAreAdvertisedOnlyWhenTheEngineTakesThem(t *testing.T) {
	old := unbookedServed
	t.Cleanup(func() { unbookedServed = old })

	unbookedServed = func() bool { return true }
	if f := identity(t).Features; !slices.Contains(f, api.FeatureContextWindowUnbooked) {
		t.Fatalf("features %v lack %s", f, api.FeatureContextWindowUnbooked)
	}
	unbookedServed = func() bool { return false }
	if f := identity(t).Features; slices.Contains(f, api.FeatureContextWindowUnbooked) {
		t.Fatalf("an engine without num_ctx_hold advertised %s", api.FeatureContextWindowUnbooked)
	}
}

// Contract v3: the line the engine planted reaches the client on a chunk of
// its own, before the text that holds it, and the done chunk lists every
// line; a non-streamed answer has only the list, as the engine's has.
func TestAV3NoticeReachesTheClientBeforeItsLine(t *testing.T) {
	line := api.Notice{Index: 0, Phase: "answer", Channel: "content", Text: "I'm writing the conclusion now.", Break: "\n"}
	mock := &mockRunner{}
	mock.CompletionFn = func(ctx context.Context, r llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
		fn(llm.CompletionResponse{Content: "Body.\n"})
		fn(llm.CompletionResponse{Notice: &line})
		fn(llm.CompletionResponse{Content: line.Text + line.Break})
		fn(llm.CompletionResponse{Done: true, DoneReason: llm.DoneReasonStop, NoticesPlanted: 1, Notices: []api.Notice{line}})
		return nil
	}
	gin.SetMode(gin.TestMode)
	s := newServerWithMockRunner(t, mock)
	createMinimalGGUFModel(t, s, "nt3", nil, "{{ .Prompt }}", nil)
	notices := json.RawMessage(`{"version":3,"answer":"default"}`)

	w := createRequest(t, s.ChatHandler, api.ChatRequest{Model: "nt3", Messages: []api.Message{{Role: "user", Content: "x"}}, Notices: notices})
	if w.Code != http.StatusOK {
		t.Fatalf("status %d %s", w.Code, w.Body)
	}
	var chunks []api.ChatResponse
	for _, l := range strings.Split(strings.TrimSpace(w.Body.String()), "\n") {
		var c api.ChatResponse
		if err := json.Unmarshal([]byte(l), &c); err != nil {
			t.Fatalf("%q: %v", l, err)
		}
		chunks = append(chunks, c)
	}
	at := slices.IndexFunc(chunks, func(c api.ChatResponse) bool { return c.Notice != nil })
	text := slices.IndexFunc(chunks, func(c api.ChatResponse) bool { return strings.Contains(c.Message.Content, line.Text) })
	if at < 0 || text < 0 || at >= text || *chunks[at].Notice != line || chunks[at].Message.Content != "" {
		t.Fatalf("notice chunk %d, its line on chunk %d, want the notice alone and first: %+v", at, text, chunks)
	}
	done := chunks[len(chunks)-1]
	if !done.Done || done.NoticesPlanted != 1 || len(done.Notices) != 1 || done.Notices[0] != line {
		t.Fatalf("done chunk = %+v, want the list", done)
	}

	stream := false
	w = createRequest(t, s.ChatHandler, api.ChatRequest{Model: "nt3", Messages: []api.Message{{Role: "user", Content: "x"}}, Notices: notices, Stream: &stream})
	var resp api.ChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Notice != nil || len(resp.Notices) != 1 || resp.Message.Content != "Body.\n"+line.Text+line.Break {
		t.Fatalf("non-streamed = %+v, want the whole text and only the list", resp)
	}
}

// generation_notice_v3 goes out only beside v2, and only for an engine that
// takes it.
func TestNoticesV3IsAdvertisedOnlyBesideV2(t *testing.T) {
	old := noticesV3Served
	t.Cleanup(func() { noticesV3Served = old })
	for _, c := range []struct{ v2, v3, want bool }{{true, true, true}, {true, false, false}, {false, true, false}} {
		withNoticesServed(t, c.v2)
		noticesV3Served = func() bool { return c.v3 }
		if got := slices.Contains(identity(t).Features, api.FeatureGenerationNoticeV3); got != c.want {
			t.Errorf("v2 %v v3 %v: advertised %v, want %v", c.v2, c.v3, got, c.want)
		}
	}
}
