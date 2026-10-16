package api

import (
	"encoding/json"

	"github.com/ollama/ollama/types/xollama"
)

// xollama-hook: notices — see docs/xollama/notices.mdx.

// XollamaNoticePath is where a client says that a message is waiting for a
// running generation (opencoti's POST /v1/notice, by the request's key).
const XollamaNoticePath = "/api/xollama/notice"

// FeatureGenerationNotice: the server passes a chat request's `notices` to an
// engine that plants generation notices (opencoti contract v2), starts it
// with the model's templates, and serves XollamaNoticePath. Listed with
// GenerationNotice only while the engine this server runs declares it.
const FeatureGenerationNotice = "generation_notice_v2"

// NoticeKindMessageWaiting is the one kind of waiting call the contract has.
const NoticeKindMessageWaiting = "message_waiting"

// GenerationNotice is the block a client reads to opt in and to hide the
// planted notices. A notice starts at a line start with a span's Open and
// runs through its Close, or to the end of its line when Close is null. It
// stays in the content; the client hides it for display and keeps it in
// history, so the prompt cache holds. The templates a request sends itself
// are hidden by that client from its own copy.
//
// On /api/xollama it is the server's default; on /api/show the model's own,
// with the templates it starts with.
type GenerationNotice struct {
	Version   int                  `json:"version"`
	Hide      []xollama.NoticeSpan `json:"hide"`
	Templates json.RawMessage      `json:"templates,omitempty"`
}

// NoticeRequest is the body of a waiting call: the key the client put in the
// running request's `notices`, and the kind. No message text reaches the model.
type NoticeRequest struct {
	Key  string `json:"key"`
	Kind string `json:"kind"`
}
