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
// call) was held, and a run of held steps longer than heldBriefly is reported
// as it is. That is the same for a built-in parser, the template's tool
// parser, harmony and a thinking-tag parser, so there is no model family it
// behaves differently for.
//
// A run that ends without a tool call (the parser gave the text out after
// all, or the response ended inside it) is closed with Released, so a client
// that opened something for it can drop it.
//
// It is nil, and does nothing, unless the request streams and asks.
type toolCallProgress struct {
	model  string
	syntax string

	// Read and written by the watch goroutine alone: every step and every
	// chunk reaches it through the one channel, in the order they happened.
	pending   string // the latest step's raw text, not yet known held or delivered
	delivered bool   // something was delivered since that step arrived
	open      bool   // a run of held steps has been reported to the client
	index     int    // of the last reported run, from 0
	tokens    int    // held steps in the current run, reported or not
	brief     string // the text of a run not yet longer than heldBriefly
}

// heldBriefly is how many steps a parser holds text that is not a tool call:
// a tag it has not finished reading, the white space before one, the tag
// that opens or closes thinking. A run is reported once it is longer.
//
// Measured on 2026-10-12 with every held step reported (gemma4:e4b,
// qwen3.5:2b, qwen2.5:1.5b and mistral-small3.1, four prompts each,
// /srv/ml/xc11/rel1/hold): 305 runs that were not a call lasted 1 to 3 steps,
// and the 12 calls 14 to 263. Reporting each of the short ones made a client
// open a tool call hundreds of times in one reply.
const heldBriefly = 8

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

// settle counts the pending step into the run when it was held, and returns
// a progress chunk once the run is longer than heldBriefly: first everything
// held so far in one chunk, then each step as it comes.
func (t *toolCallProgress) settle() (api.ChatResponse, bool) {
	if t.delivered {
		return api.ChatResponse{}, false
	}
	t.delivered = true
	t.tokens++
	text := t.pending
	if !t.open {
		t.brief += text
		if t.tokens <= heldBriefly {
			return api.ChatResponse{}, false
		}
		t.open = true
		t.index++
		text, t.brief = t.brief, ""
	}
	return api.ChatResponse{
		Model:     t.model,
		CreatedAt: time.Now().UTC(),
		Message: api.Message{Role: "assistant", ToolCallProgress: &api.ToolCallProgress{
			Index: t.index, Syntax: t.syntax, Text: text, Tokens: t.tokens,
		}},
	}, true
}

// deliver sees a chunk the handler is sending. Content, thinking, a tool call
// or the end of the response means the pending step was not held, and ends
// the run. A reported run is closed on that chunk: Done when it carries the
// tool calls, Released when it does not.
func (t *toolCallProgress) deliver(res api.ChatResponse) api.ChatResponse {
	m := res.Message
	if m.Content == "" && m.Thinking == "" && len(m.ToolCalls) == 0 && !res.Done {
		return res // a chunk of logprobs alone delivers no text
	}
	t.delivered = true
	if t.open {
		called := len(m.ToolCalls) > 0
		res.Message.ToolCallProgress = &api.ToolCallProgress{Index: t.index, Syntax: t.syntax, Tokens: t.tokens, Done: called, Released: !called}
	}
	t.open, t.tokens, t.brief = false, 0, ""
	return res
}
