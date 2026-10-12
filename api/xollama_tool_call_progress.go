package api

// ToolCallProgress is model output the server is holding back, reported
// while it is held (tool_call_progress_v1). A tool call is parsed only when
// it is whole, so without this a client sees nothing between the call's first
// token and its last. It is sent on streamed chat chunks to a client that
// asked with ChatRequest.ToolCallProgress, and never replaces the call itself:
// that still arrives in Message.ToolCalls, parsed, when it is complete.
//
// The rule is one for every model: a step of engine output for which the
// server delivered nothing (no content, no thinking, no tool call) is held,
// and its raw text is reported.
type ToolCallProgress struct {
	// Index numbers the runs of held output in this response from 0. A run
	// ends when the server delivers something.
	Index int `json:"index"`
	// Syntax names how Text is written: the model's parser ("gemma4",
	// "qwen3.5", ...), or "template" for a model without one. Text is the
	// model's own notation, not JSON, unless that is what the model writes.
	Syntax string `json:"syntax"`
	// Text is the raw output of one held step, delimiters included, exactly
	// as the model wrote it. A step may end anywhere, mid-token included.
	Text string `json:"text"`
	// Tokens counts the held steps of this run so far: close to its tokens,
	// not a tokenizer count.
	Tokens int `json:"tokens"`
	// Done is set, with an empty Text, on the chunk that delivers the tool
	// calls a run was held for.
	Done bool `json:"done,omitempty"`
}
