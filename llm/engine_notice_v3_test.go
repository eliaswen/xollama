package llm

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
)

// v3Engine streams one route the way opencoti c13 r5 does (mails #1576-#1582):
// a top-level `notice` on the chunk of the line's first token, which on the
// chat route may carry no text yet (the chat parser held it), and the list in
// the final chunk's timings.
func v3Engine(t *testing.T, path string, lines ...string) *llamaServerRunner {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			fmt.Fprint(w, `{"status":"ok"}`)
		case "/props":
			fmt.Fprint(w, `{"features":["generation_notice_v2","generation_notice_v3"]}`)
		case path:
			w.Header().Set("Content-Type", "text/event-stream")
			for _, l := range lines {
				fmt.Fprintln(w, "data: "+l)
				fmt.Fprintln(w)
			}
		}
	}))
	t.Cleanup(srv.Close)
	parts := strings.Split(srv.URL, ":")
	var port int
	fmt.Sscanf(parts[len(parts)-1], "%d", &port)
	return &llamaServerRunner{
		port:         port,
		cmd:          fakeRunningCmd(),
		sem:          semaphore.NewWeighted(1),
		options:      api.Options{Runner: api.Runner{NumCtx: 2048}},
		usedOpencoti: true,
	}
}

const v3Line = `{"index":0,"phase":"answer","channel":"content","text":"I'm writing the conclusion now.","break":"\n"}`

func wantV3Notice(t *testing.T, got *api.Notice) {
	t.Helper()
	want := api.Notice{Index: 0, Phase: "answer", Channel: "content", Text: "I'm writing the conclusion now.", Break: "\n"}
	if got == nil || *got != want {
		t.Fatalf("notice = %+v, want %+v", got, want)
	}
}

// On /completion the notice and the line's first character share a chunk; a
// chunk that carries only the notice is still handed on, not dropped with the
// empty text it came with.
func TestAV3NoticeOnCompletionReachesTheCallerWithItsList(t *testing.T) {
	runner := v3Engine(t, "/completion",
		`{"content":"Body.\n","stop":false}`,
		`{"content":"","stop":false,"notice":`+v3Line+`}`,
		`{"content":"I'm writing the conclusion now.\n","stop":false}`,
		`{"content":"","stop":true,"stop_type":"eos","timings":{"predicted_n":3,"notices_planted":1,"notices":[`+v3Line+`]}}`,
	)
	opts := api.DefaultOptions()
	var got []CompletionResponse
	if err := runner.Completion(t.Context(), CompletionRequest{Prompt: "p", Options: &opts}, func(r CompletionResponse) { got = append(got, r) }); err != nil {
		t.Fatal(err)
	}
	if len(got) != 4 {
		t.Fatalf("got %d responses, want 4: %+v", len(got), got)
	}
	if got[0].Notice != nil || got[2].Notice != nil {
		t.Fatalf("a notice on a chunk the engine sent none with: %+v", got)
	}
	wantV3Notice(t, got[1].Notice)
	last := got[3]
	if !last.Done || last.NoticesPlanted != 1 || len(last.Notices) != 1 {
		t.Fatalf("done = %+v, want the planted count and the list", last)
	}
	wantV3Notice(t, &last.Notices[0])
}

func TestAV3NoticeOnChatReachesTheCallerWithItsList(t *testing.T) {
	runner := v3Engine(t, "/v1/chat/completions",
		`{"choices":[{"delta":{"content":"Body.\n"}}]}`,
		`{"choices":[{"delta":{}}],"notice":`+v3Line+`}`,
		`{"choices":[{"delta":{"content":"I'm writing the conclusion now.\n"}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"predicted_n":3,"notices_planted":1,"notices":[`+v3Line+`]}}`,
		`[DONE]`,
	)
	opts := api.DefaultOptions()
	var got []ChatResponse
	if err := runner.Chat(t.Context(), ChatRequest{Messages: []api.Message{{Role: "user", Content: "hi"}}, Options: &opts}, func(r ChatResponse) { got = append(got, r) }); err != nil {
		t.Fatal(err)
	}
	if len(got) != 4 {
		t.Fatalf("got %d responses, want 4: %+v", len(got), got)
	}
	wantV3Notice(t, got[1].Notice)
	last := got[3]
	if !last.Done || last.NoticesPlanted != 1 || len(last.Notices) != 1 {
		t.Fatalf("done = %+v, want the planted count and the list", last)
	}
	wantV3Notice(t, &last.Notices[0])
}

// A v2 engine sends no `notice`: nothing changes for its stream.
func TestAV2StreamCarriesNoNotice(t *testing.T) {
	runner := v3Engine(t, "/completion",
		`{"content":"<engine_status>Time to conclude.</engine_status>\n","stop":false}`,
		`{"content":"","stop":true,"stop_type":"eos","timings":{"notices_planted":1}}`,
	)
	opts := api.DefaultOptions()
	var got []CompletionResponse
	if err := runner.Completion(t.Context(), CompletionRequest{Prompt: "p", Options: &opts}, func(r CompletionResponse) { got = append(got, r) }); err != nil {
		t.Fatal(err)
	}
	for _, r := range got {
		if r.Notice != nil || r.Notices != nil {
			t.Fatalf("a v2 stream produced v3 records: %+v", r)
		}
	}
}
