package llm

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
)

// The engine's own event on the Mac mini (opencoti bug-4048, temperature 0).
const vectorEvent = `data: {"error":{"code":500,"message":"vector","type":"server_error"}}`

// streamed runs one completion against an engine that answers with lines.
func streamed(t *testing.T, opencoti bool, lines ...string) ([]CompletionResponse, error) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			fmt.Fprint(w, `{"status":"ok"}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, line := range lines {
			fmt.Fprintln(w, line)
			fmt.Fprintln(w)
		}
	}))
	t.Cleanup(srv.Close)
	u, _ := url.Parse(srv.URL)
	port, _ := strconv.Atoi(u.Port())
	runner := &llamaServerRunner{
		port:         port,
		cmd:          fakeRunningCmd(),
		sem:          semaphore.NewWeighted(1),
		options:      api.Options{Runner: api.Runner{NumCtx: 2048}},
		usedOpencoti: opencoti,
	}
	var got []CompletionResponse
	opts := api.DefaultOptions()
	err := runner.Completion(t.Context(), CompletionRequest{Prompt: "hi", Options: &opts}, func(cr CompletionResponse) {
		got = append(got, cr)
	})
	return got, err
}

func TestAnOpencotiErrorEventFailsTheCompletion(t *testing.T) {
	for name, lines := range map[string][]string{
		"before any token": {vectorEvent},
		"after a token":    {`data: {"content":"Hello","stop":false}`, vectorEvent},
		"as plain text":    {`data: {"error":"slot unavailable"}`},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := streamed(t, true, lines...)
			if err == nil {
				t.Fatalf("no error; the client would get an empty 200 (%d responses)", len(got))
			}
			want := "vector"
			if name == "as plain text" {
				want = "slot unavailable"
			}
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error %q does not carry the engine's message %q", err, want)
			}
			for _, r := range got {
				if r.Done {
					t.Errorf("a failed completion was reported done: %+v", r)
				}
			}
		})
	}
}

func TestAnOpencotiStreamWithoutItsFinalEventFailsTheCompletion(t *testing.T) {
	for name, lines := range map[string][]string{
		"nothing at all": nil,
		"tokens only":    {`data: {"content":"Hello","stop":false}`},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := streamed(t, true, lines...); err == nil {
				t.Fatal("no error for a stream that never finished")
			}
		})
	}
}

func TestAFinishedOpencotiCompletionIsNotAnError(t *testing.T) {
	got, err := streamed(t, true,
		`data: {"content":"Hello","stop":false}`,
		`data: {"content":"","stop":true,"stop_type":"eos","error":null}`)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || !got[1].Done {
		t.Fatalf("got %+v, want a token and a done", got)
	}
}

// Off means off: stock llama.cpp keeps upstream's reading of both streams.
func TestStockLlamaCppKeepsUpstreamsStreamHandling(t *testing.T) {
	for name, lines := range map[string][]string{
		"error event":    {vectorEvent},
		"no final event": {`data: {"content":"Hello","stop":false}`},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := streamed(t, false, lines...); err != nil {
				t.Fatalf("stock llama.cpp returned %v, upstream returns nil", err)
			}
		})
	}
}
