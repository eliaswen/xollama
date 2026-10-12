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
// and a run of held steps is reported, raw, once it is longer than the few
// steps a parser holds text that is not a call.
type ToolCallProgress struct {
	// Index numbers the reported runs of held output in this response from
	// 0. A run ends when the server delivers something.
	Index int `json:"index"`
	// Syntax names how Text is written: the model's parser ("gemma4",
	// "qwen3.5", ...), or "template" for a model without one. Text is the
	// model's own notation, not JSON, unless that is what the model writes.
	Syntax string `json:"syntax"`
	// Text is raw held output, delimiters included, exactly as the model
	// wrote it: the steps held so far on a run's first chunk, one step on each
	// later one. A step may end anywhere, mid-token included. A run starts
	// where the server began holding, which can be before the call: the tag
	// that closes the model's thinking, or white space.
	Text string `json:"text"`
	// Tokens counts the held steps of this run so far: close to its tokens,
	// not a tokenizer count.
	Tokens int `json:"tokens"`
	// Done is set, with an empty Text, on the chunk that delivers the tool
	// calls a run was held for.
	Done bool `json:"done,omitempty"`
	// Released is set, with an empty Text, on the chunk that ends a reported
	// run without a tool call: the text was given out as content or thinking
	// after all, or the response ended inside it (a call cut off by the token
	// limit). Whatever a client opened for the run is to be dropped.
	Released bool `json:"released,omitempty"`
}
