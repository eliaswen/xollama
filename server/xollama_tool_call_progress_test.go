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

// held joins the progress text of one run and reports how it ended.
func held(t *testing.T, chunks []api.ChatResponse, syntax string) (text string, steps int, sawDone bool, calls int) {
	t.Helper()
	var b strings.Builder
	for _, c := range chunks {
		p := c.Message.ToolCallProgress
		if p != nil {
			if p.Syntax != syntax || p.Index != 0 {
				t.Fatalf("step = %+v, want syntax %s, index 0", p, syntax)
			}
			if calls > 0 {
				t.Fatalf("progress arrived after the parsed call: %+v", p)
			}
			if p.Done {
				sawDone = true
			} else {
				steps++
				if p.Tokens != steps {
					t.Fatalf("tokens = %d on held step %d", p.Tokens, steps)
				}
			}
			b.WriteString(p.Text)
		}
		calls += len(c.Message.ToolCalls)
	}
	return b.String(), steps, sawDone, calls
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
			pieces: []string{"On it.", "<|tool_call>", "call:edit{intent:<|\"|>fix", " the loop<|\"|>,", "path:<|\"|>a.go<|\"|>}", "<tool_call|>", ""},
			want:   "<|tool_call>call:edit{intent:<|\"|>fix the loop<|\"|>,path:<|\"|>a.go<|\"|>}",
		},
		{
			name:   "a built-in parser that does not",
			create: api.CreateRequest{Renderer: "qwen3-coder", Parser: "qwen3-coder"},
			syntax: "qwen3-coder",
			pieces: []string{"On it.", "<tool_call>", "<function=edit>", "<parameter=intent>fix the loop</parameter>", "<parameter=path>a.go</parameter>", "</function>", "</tool_call>", ""},
			want:   "<tool_call><function=edit><parameter=intent>fix the loop</parameter><parameter=path>a.go</parameter></function>",
		},
		{
			name:   "no built-in parser, the template's tool calls",
			create: api.CreateRequest{Template: templ},
			syntax: "template",
			pieces: []string{`{"name":`, ` "edit",`, ` "arguments": {"intent":`, ` "fix the loop",`, ` "path": "a.go"}`, `}`, ""},
			want:   `{"name": "edit", "arguments": {"intent": "fix the loop", "path": "a.go"}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := progressServer(t, tc.create, tc.pieces)
			text, steps, sawDone, calls := held(t, progressChat(t, s, true, true), tc.syntax)

			if text != tc.want {
				t.Fatalf("progress text = %q, want %q", text, tc.want)
			}
			if steps < 4 {
				t.Fatalf("steps = %d: the call must be reported piece by piece, not once at the end", steps)
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

var progressPieces = []string{
	"On it.",
	"<|tool_call>",
	"call:edit{intent:<|\"|>fix",
	" the loop<|\"|>,",
	"path:<|\"|>a.go<|\"|>}",
	"<tool_call|>",
	"",
}

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

func TestTheIdentityRouteAnnouncesToolCallProgress(t *testing.T) {
	for _, f := range xollamaFeatures() {
		if f == api.FeatureToolCallProgress {
			return
		}
	}
	t.Fatalf("%s is not announced", api.FeatureToolCallProgress)
}
