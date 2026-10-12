package server

import (
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/parsers"
)

// toolCallProgress reports a tool call to the client while the model is still
// writing it (tool_call_progress_v1). The parser holds a call back until its
// closing delimiter, which for a call that runs away is minutes of silence
// and tens of thousands of tokens nobody can stop; with this a client can
// read the call as it grows and cancel the request.
//
// It is off unless the request asks, so a client that does not know the field
// gets the response it always got.
type toolCallProgress struct {
	follow *parsers.CallProgress
	syntax string
	model  string
}

// newToolCallProgress returns nil, which reports nothing, unless the request
// streams, asks for progress and is parsed by a parser that names its
// tool-call delimiters.
func newToolCallProgress(req *api.ChatRequest, p parsers.Parser, parserName string) *toolCallProgress {
	if !req.ToolCallProgress || (req.Stream != nil && !*req.Stream) {
		return nil
	}
	follow := parsers.NewCallProgress(p)
	if follow == nil {
		return nil
	}
	return &toolCallProgress{follow: follow, syntax: parserName, model: req.Model}
}

// attach reads the next piece of raw engine text and puts the step it holds
// on res, reporting whether there was one. When a piece holds two steps (a
// call closing and the next opening), the earlier goes out first on a chunk
// of its own.
func (t *toolCallProgress) attach(res *api.ChatResponse, raw string, ch chan<- any) bool {
	if t == nil {
		return false
	}
	deltas := t.follow.Add(raw)
	if len(deltas) == 0 {
		return false
	}
	for _, d := range deltas[:len(deltas)-1] {
		ch <- api.ChatResponse{
			Model:     t.model,
			CreatedAt: time.Now().UTC(),
			Message:   api.Message{Role: "assistant", ToolCallProgress: t.step(d)},
		}
	}
	res.Message.ToolCallProgress = t.step(deltas[len(deltas)-1])
	return true
}

func (t *toolCallProgress) step(d parsers.CallDelta) *api.ToolCallProgress {
	return &api.ToolCallProgress{Index: d.Index, Syntax: t.syntax, Text: d.Text, Tokens: d.Tokens, Done: d.Done}
}
