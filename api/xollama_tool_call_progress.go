package api

// ToolCallProgress is what the model has written of a tool call that is not
// complete yet (tool_call_progress_v1). The server parses a call only when it
// is whole, so without this a client sees nothing between the call's first
// token and its last. It is sent on streamed chat chunks to a client that
// asked with ChatRequest.ToolCallProgress, and never replaces the call itself:
// that still arrives in Message.ToolCalls, parsed, when it is complete.
type ToolCallProgress struct {
	// Index numbers the calls of this response from 0.
	Index int `json:"index"`
	// Syntax names how Text is written: the model's parser ("gemma4",
	// "qwen3.5"). Text is the model's own notation, not JSON.
	Syntax string `json:"syntax"`
	// Text is the raw text added since the previous chunk for this call. It
	// starts at the first character after the call's opening delimiter and
	// never contains a delimiter. It may be empty on the chunk that opens the
	// call, and a chunk may end anywhere, mid-token included.
	Text string `json:"text"`
	// Tokens counts the engine's output pieces this call has taken so far,
	// its delimiters included: close to its tokens, not a tokenizer count.
	Tokens int `json:"tokens"`
	// Done is set on the chunk that reads the call's closing delimiter.
	Done bool `json:"done,omitempty"`
}
