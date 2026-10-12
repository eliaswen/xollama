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

// progressServer serves one model parsed by gemma4 whose engine writes a tool
// call in the given pieces.
func progressServer(t *testing.T, pieces []string) *Server {
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
	if w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:    "progress",
		Files:    map[string]string{"file.gguf": digest},
		Renderer: "gemma4",
		Parser:   "gemma4",
		Stream:   &noStream,
	}); w.Code != http.StatusOK {
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

var progressPieces = []string{
	"On it.",
	"<|tool_call>",
	"call:edit{intent:<|\"|>fix",
	" the loop<|\"|>,",
	"path:<|\"|>a.go<|\"|>}",
	"<tool_call|>",
	"",
}

func TestAToolCallIsReportedWhileItIsWritten(t *testing.T) {
	s := progressServer(t, progressPieces)
	chunks := progressChat(t, s, true, true)

	var text strings.Builder
	var steps, calls int
	var sawDone bool
	for _, c := range chunks {
		calls += len(c.Message.ToolCalls)
		p := c.Message.ToolCallProgress
		if p == nil {
			continue
		}
		if calls > 0 && len(c.Message.ToolCalls) == 0 {
			t.Fatalf("progress arrived after the parsed call: %+v", p)
		}
		steps++
		if p.Syntax != "gemma4" || p.Index != 0 {
			t.Fatalf("step = %+v, want syntax gemma4, index 0", p)
		}
		text.WriteString(p.Text)
		sawDone = sawDone || p.Done
	}

	if want := "call:edit{intent:<|\"|>fix the loop<|\"|>,path:<|\"|>a.go<|\"|>}"; text.String() != want {
		t.Fatalf("progress text = %q, want %q", text.String(), want)
	}
	if steps < 4 {
		t.Fatalf("steps = %d: the call must be reported piece by piece, not once at the end", steps)
	}
	if !sawDone {
		t.Fatal("no step said the call was closed")
	}
	if calls != 1 {
		t.Fatalf("parsed tool calls = %d, want the one call, whole, as without progress", calls)
	}
}

func TestToolCallProgressIsOffUnlessAsked(t *testing.T) {
	s := progressServer(t, progressPieces)

	for _, c := range progressChat(t, s, false, true) {
		if c.Message.ToolCallProgress != nil {
			t.Fatalf("a request that did not ask got progress: %+v", c.Message.ToolCallProgress)
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

func TestTheIdentityRouteAnnouncesToolCallProgress(t *testing.T) {
	for _, f := range xollamaFeatures() {
		if f == api.FeatureToolCallProgress {
			return
		}
	}
	t.Fatalf("%s is not announced", api.FeatureToolCallProgress)
}
