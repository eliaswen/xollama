package server

import (
	"time"

	"github.com/ollama/ollama/api"
)

// toolCallProgress reports what the model has written while the server is
// holding it back (tool_call_progress_v1). A tool call is parsed only when it
// is whole, so a call that runs away is minutes of silence and tens of
// thousands of tokens nobody can stop; with this a client can read the call
// as it grows and cancel the request.
//
// It knows nothing about any parser. The one rule: a step of engine output
// for which the handler delivered nothing (no content, no thinking, no tool
// call) was held, and held text is reported as it is. That is the same for a
// built-in parser, the template's tool parser, harmony and a thinking-tag
// parser, so there is no model family it behaves differently for.
//
// It is nil, and does nothing, unless the request streams and asks.
type toolCallProgress struct {
	model  string
	syntax string

	// Read and written by the watch goroutine alone: every step and every
	// chunk reaches it through the one channel, in the order they happened.
	pending   string // the latest step's raw text, not yet known held or delivered
	delivered bool   // something was delivered since that step arrived
	open      bool   // a run of held steps is being reported
	index     int    // of that run, from 0
	tokens    int    // held steps in it
}

// templateSyntax names the notation of a model without a built-in parser:
// whatever its template makes it write, usually JSON.
const templateSyntax = "template"

// progressStep is one piece of raw engine output, put on the response channel
// ahead of whatever the handler sends for it. It never reaches the client.
type progressStep struct{ raw string }

func newToolCallProgress(req *api.ChatRequest, parserName string) *toolCallProgress {
	if !req.ToolCallProgress || (req.Stream != nil && !*req.Stream) {
		return nil
	}
	if parserName == "" {
		parserName = templateSyntax
	}
	return &toolCallProgress{model: req.Model, syntax: parserName, index: -1, delivered: true}
}

// step announces the next piece of engine output, before the handler has
// decided what to do with it.
func (t *toolCallProgress) step(ch chan<- any, raw string) {
	if t != nil {
		ch <- progressStep{raw}
	}
}

// watch returns the channel the response is written from: ch itself when
// there is nothing to report, else ch with the held steps added in order.
func (t *toolCallProgress) watch(ch chan any) chan any {
	if t == nil {
		return ch
	}
	out := make(chan any)
	go func() {
		defer close(out)
		for v := range ch {
			switch v := v.(type) {
			case progressStep:
				// The step before this one is settled now: nothing was
				// delivered for it, so it was held.
				if held, ok := t.settle(); ok {
					out <- held
				}
				t.pending, t.delivered = v.raw, v.raw == ""
			case api.ChatResponse:
				out <- t.deliver(v)
			default:
				out <- v
			}
		}
	}()
	return out
}

// settle turns the pending step into a progress chunk when it was held.
func (t *toolCallProgress) settle() (api.ChatResponse, bool) {
	if t.delivered {
		return api.ChatResponse{}, false
	}
	if !t.open {
		t.open = true
		t.index++
		t.tokens = 0
	}
	t.tokens++
	t.delivered = true
	return api.ChatResponse{
		Model:     t.model,
		CreatedAt: time.Now().UTC(),
		Message: api.Message{Role: "assistant", ToolCallProgress: &api.ToolCallProgress{
			Index: t.index, Syntax: t.syntax, Text: t.pending, Tokens: t.tokens,
		}},
	}, true
}

// deliver sees a chunk the handler is sending. Content, thinking, a tool call
// or the end of the response means the pending step was not held, and ends
// the run being reported; a chunk that carries tool calls says so with Done.
func (t *toolCallProgress) deliver(res api.ChatResponse) api.ChatResponse {
	m := res.Message
	if m.Content == "" && m.Thinking == "" && len(m.ToolCalls) == 0 && !res.Done {
		return res // a chunk of logprobs alone delivers no text
	}
	t.delivered = true
	if t.open {
		t.open = false
		if len(m.ToolCalls) > 0 {
			res.Message.ToolCallProgress = &api.ToolCallProgress{Index: t.index, Syntax: t.syntax, Tokens: t.tokens, Done: true}
		}
	}
	return res
}
