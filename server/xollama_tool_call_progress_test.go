package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// progressServer serves one model, made from create, whose engine writes the
// given pieces.
func progressServer(t *testing.T, create api.CreateRequest, pieces []string) *Server {
	t.Helper()
	gin.SetMode(gin.TestMode)

	mock := mockRunner{}
	mock.CompletionFn = func(ctx context.Context, r llm.CompletionRequest, fn func(r llm.CompletionResponse)) error {
		for i, p := range pieces {
			resp := llm.CompletionResponse{Content: p}
			if i == len(pieces)-1 {
				resp.Done = true
				resp.DoneReason = llm.DoneReasonStop
			}
			fn(resp)
		}
		return nil
	}

	s := Server{
		sched: &Scheduler{
			pendingReqCh:    make(chan *LlmRequest, 1),
			finishedReqCh:   make(chan *LlmRequest, 1),
			expiredCh:       make(chan *runnerRef, 1),
			unloadedCh:      make(chan any, 1),
			loaded:          make(map[string]*runnerRef),
			newServerFn:     newMockServer(&mock),
			getGpuFn:        getGpuFn,
			getSystemInfoFn: getSystemInfoFn,
			waitForRecovery: 250 * time.Millisecond,
			loadFn: func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
				req.successCh <- &runnerRef{llama: &mock}
				return false
			},
		},
	}
	go s.sched.Run(t.Context())

	_, digest := createBinFile(t, gguftest.KV{
		"general.architecture":          "llama",
		"llama.block_count":             uint32(1),
		"llama.context_length":          uint32(8192),
		"llama.embedding_length":        uint32(4096),
		"llama.attention.head_count":    uint32(32),
		"llama.attention.head_count_kv": uint32(8),
		"tokenizer.ggml.tokens":         []string{""},
		"tokenizer.ggml.scores":         []float32{0},
		"tokenizer.ggml.token_type":     []int32{0},
	}, []*gguftest.Tensor{
		{Name: "token_embd.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.attn_norm.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.ffn_down.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.ffn_gate.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.ffn_up.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.ffn_norm.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.attn_k.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.attn_output.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.attn_q.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "blk.0.attn_v.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
		{Name: "output.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
	})

	noStream := false
	create.Model = "progress"
	create.Files = map[string]string{"file.gguf": digest}
	create.Stream = &noStream
	if w := createRequest(t, s.CreateHandler, create); w.Code != http.StatusOK {
		t.Fatalf("create: %d: %s", w.Code, w.Body.String())
	}
	return &s
}

func progressChat(t *testing.T, s *Server, ask bool, stream bool) []api.ChatResponse {
	t.Helper()
	think := false
	w := createRequest(t, s.ChatHandler, api.ChatRequest{
		Model:    "progress",
		Messages: []api.Message{{Role: "user", Content: "fix the loop"}},
		Tools: []api.Tool{{Type: "function", Function: api.ToolFunction{
			Name: "edit",
			Parameters: api.ToolFunctionParameters{Type: "object", Properties: func() *api.ToolPropertiesMap {
				m := api.NewToolPropertiesMap()
				m.Set("intent", api.ToolProperty{Type: api.PropertyType{"string"}})
				m.Set("path", api.ToolProperty{Type: api.PropertyType{"string"}})
				return m
			}()},
		}}},
		Think:            &api.ThinkValue{Value: think},
		Stream:           &stream,
		ToolCallProgress: ask,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("chat: %d: %s", w.Code, w.Body.String())
	}
	var out []api.ChatResponse
	dec := json.NewDecoder(w.Body)
	for dec.More() {
		var r api.ChatResponse
		if err := dec.Decode(&r); err != nil {
			t.Fatalf("decode: %v", err)
		}
		out = append(out, r)
	}
	if len(out) == 0 || !out[len(out)-1].Done {
		t.Fatalf("the stream did not end with done: %+v", out)
	}
	return out
}

// small cuts s into pieces of a few characters, as an engine's tokens are.
func small(s string) []string {
	var out []string
	for len(s) > 5 {
		out, s = append(out, s[:5]), s[5:]
	}
	return append(out, s)
}

// written is what an engine writes: said whole, then written in small pieces,
// then the empty last step.
func written(said, text string) []string {
	return append(append([]string{said}, small(text)...), "")
}

// held joins the progress text of the one reported run and says how it
// ended.
func held(t *testing.T, chunks []api.ChatResponse, syntax string) (text string, tokens int, sawDone, sawReleased bool, calls int) {
	t.Helper()
	var b strings.Builder
	first := true
	for _, c := range chunks {
		p := c.Message.ToolCallProgress
		if p != nil {
			if p.Syntax != syntax || p.Index != 0 {
				t.Fatalf("step = %+v, want syntax %s, index 0", p, syntax)
			}
			if calls > 0 || sawDone || sawReleased {
				t.Fatalf("progress arrived after the run ended: %+v", p)
			}
			switch {
			case p.Done || p.Released:
				sawDone, sawReleased = p.Done, p.Released
				if p.Text != "" || p.Tokens != tokens {
					t.Fatalf("the closing chunk = %+v, want no text and the run's %d tokens", p, tokens)
				}
			case first:
				// The run is reported once it is longer than a brief hold,
				// with everything held until then.
				if p.Tokens != heldBriefly+1 {
					t.Fatalf("the first report has tokens = %d, want %d", p.Tokens, heldBriefly+1)
				}
				first, tokens = false, p.Tokens
			default:
				tokens++
				if p.Tokens != tokens {
					t.Fatalf("tokens = %d on held step %d", p.Tokens, tokens)
				}
			}
			b.WriteString(p.Text)
		}
		calls += len(c.Message.ToolCalls)
	}
	return b.String(), tokens, sawDone, sawReleased, calls
}

// Every way a model's output is parsed reports a held call the same way: the
// raw text of the steps nothing was delivered for. One case per kind of
// parser, and none of them is named in the code under test.
func TestAToolCallIsReportedWhileItIsWritten(t *testing.T) {
	templ := `
{{- if .Tools }}
{{ .Tools }}
{{ end }}
{{- range .Messages }}
{{- .Role }}: {{ .Content }}
{{- range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{- end }}
{{ end }}`

	cases := []struct {
		name   string
		create api.CreateRequest
		syntax string
		pieces []string
		want   string
	}{
		{
			name:   "a built-in parser that names its delimiters",
			create: api.CreateRequest{Renderer: "gemma4", Parser: "gemma4"},
			syntax: "gemma4",
			pieces: written("On it.", "<|tool_call>call:edit{intent:<|\"|>fix the loop<|\"|>,path:<|\"|>a.go<|\"|>}<tool_call|>"),
			want:   "<|tool_call>call:edit{intent:<|\"|>fix the loop<|\"|>,path:<|\"|>a.go<|\"|>}<tool_call|>",
		},
		{
			name:   "a built-in parser that does not",
			create: api.CreateRequest{Renderer: "qwen3-coder", Parser: "qwen3-coder"},
			syntax: "qwen3-coder",
			pieces: written("On it.", "<tool_call><function=edit><parameter=intent>fix the loop</parameter><parameter=path>a.go</parameter></function></tool_call>"),
			want:   "<tool_call><function=edit><parameter=intent>fix the loop</parameter><parameter=path>a.go</parameter></function></tool_call>",
		},
		{
			name:   "no built-in parser, the template's tool calls",
			create: api.CreateRequest{Template: templ},
			syntax: "template",
			pieces: append(small(`{"name": "edit", "arguments": {"intent": "fix the loop", "path": "a.go"}}`), ""),
			want:   `{"name": "edit", "arguments": {"intent": "fix the loop", "path": "a.go"}}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := progressServer(t, tc.create, tc.pieces)
			text, tokens, sawDone, sawReleased, calls := held(t, progressChat(t, s, true, true), tc.syntax)

			// The step that completes the call is delivered, not held, so
			// the report is the call up to its closing delimiter.
			if !strings.HasPrefix(tc.want, text) || len(text) < len(tc.want)-len("</tool_call>") {
				t.Fatalf("progress text = %q, want %q up to the step that closed the call", text, tc.want)
			}
			if tokens < heldBriefly+3 {
				t.Fatalf("tokens = %d: the call must be reported as it grows, not once at the end", tokens)
			}
			if sawReleased {
				t.Fatal("a run that ended in a call was closed as released")
			}
			if calls != 1 {
				t.Fatalf("parsed tool calls = %d, want the one call, whole, as without progress", calls)
			}
			if !sawDone {
				t.Fatal("the chunk that delivered the call did not say the held run was done")
			}
		})
	}
}

var progressPieces = written("On it.", "<|tool_call>call:edit{intent:<|\"|>fix the loop<|\"|>,path:<|\"|>a.go<|\"|>}<tool_call|>")

func TestToolCallProgressIsOffUnlessAsked(t *testing.T) {
	gemma := api.CreateRequest{Renderer: "gemma4", Parser: "gemma4"}
	s := progressServer(t, gemma, progressPieces)

	off := progressChat(t, s, false, true)
	for _, c := range off {
		if c.Message.ToolCallProgress != nil {
			t.Fatalf("a request that did not ask got progress: %+v", c.Message.ToolCallProgress)
		}
	}
	// And asking adds chunks, it changes none: what is left when the
	// progress is taken away is the response a client gets without it.
	var on []api.ChatResponse
	for _, c := range progressChat(t, s, true, true) {
		m := c.Message
		if m.ToolCallProgress != nil && m.Content == "" && m.Thinking == "" && len(m.ToolCalls) == 0 && !c.Done {
			continue
		}
		on = append(on, c)
	}
	if len(on) != len(off) {
		t.Fatalf("with progress removed %d chunks are left, without asking there are %d", len(on), len(off))
	}
	for i := range on {
		if on[i].Message.Content != off[i].Message.Content || len(on[i].Message.ToolCalls) != len(off[i].Message.ToolCalls) || on[i].Done != off[i].Done {
			t.Fatalf("chunk %d differs: %+v vs %+v", i, on[i].Message, off[i].Message)
		}
	}

	// A response that is not streamed has nobody to report to.
	out := progressChat(t, s, true, false)
	if len(out) != 1 || out[0].Message.ToolCallProgress != nil {
		t.Fatalf("an unstreamed response carries no progress: %+v", out)
	}
	if len(out[0].Message.ToolCalls) != 1 {
		t.Fatalf("the call itself must still be there: %+v", out[0].Message)
	}
}

// Text the server delivers is never also reported as held.
func TestDeliveredTextIsNotReportedAsHeld(t *testing.T) {
	s := progressServer(t, api.CreateRequest{Renderer: "gemma4", Parser: "gemma4"}, []string{"Hello", " there", ", all", " good.", ""})
	for _, c := range progressChat(t, s, true, true) {
		if p := c.Message.ToolCallProgress; p != nil {
			t.Fatalf("plain content was reported as held: %+v", p)
		}
	}
}

// A parser holds text for a few steps for reasons that are not a call: a tag
// it has not finished reading, the white space before one. None of that is
// reported: a client would open a tool call for each (hundreds in one reply
// of a thinking model, measured).
func TestABriefHoldIsNotReportedAsACall(t *testing.T) {
	cases := map[string][]string{
		"white space between lines":   {"First line.", "\n", "    ", "Second line.", "\n", "    ", "Third.", ""},
		"a tag that opens no call":    {"Compare a ", "<", "|", "b and c.", ""},
		"the start of a call's tag":   {"Use ", "<|tool", "_ca", " tags sparingly.", ""},
		"a held step as the last one": {"Done.", "\n", ""},
	}
	for name, pieces := range cases {
		t.Run(name, func(t *testing.T) {
			s := progressServer(t, api.CreateRequest{Renderer: "gemma4", Parser: "gemma4"}, pieces)
			for _, c := range progressChat(t, s, true, true) {
				if p := c.Message.ToolCallProgress; p != nil {
					t.Fatalf("a brief hold was reported: %+v", p)
				}
			}
		})
	}
}

// A call the model never closes is the case this exists for: it is reported
// while it grows, and when the response ends inside it the run is closed as
// released, because no call came of it.
func TestARunThatEndsWithoutACallIsReleased(t *testing.T) {
	open := "<tool_call><function=edit><parameter=intent>fix the loop and then the next one and the one after"
	s := progressServer(t, api.CreateRequest{Renderer: "qwen3-coder", Parser: "qwen3-coder"}, written("On it.", open))
	chunks := progressChat(t, s, true, true)
	text, tokens, sawDone, sawReleased, calls := held(t, chunks, "qwen3-coder")
	if !strings.HasPrefix(open, text) || tokens <= heldBriefly {
		t.Fatalf("progress text = %q over %d tokens, want the unclosed call as it grew", text, tokens)
	}
	if calls != 0 || sawDone {
		t.Fatalf("calls = %d, done = %v: no call was completed", calls, sawDone)
	}
	if !sawReleased {
		t.Fatal("the run ended without a call and was not closed as released")
	}
	if last := chunks[len(chunks)-1]; last.Message.ToolCallProgress == nil || !last.Message.ToolCallProgress.Released {
		t.Fatalf("the release belongs on the chunk that ended the run: %+v", last.Message)
	}
}

func TestTheIdentityRouteAnnouncesToolCallProgress(t *testing.T) {
	for _, f := range xollamaFeatures() {
		if f == api.FeatureToolCallProgress {
			return
		}
	}
	t.Fatalf("%s is not announced", api.FeatureToolCallProgress)
}
