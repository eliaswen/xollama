package api

// xollama-hook: notices — see docs/xollama/notices.mdx.

// XollamaNoticePath is where a client says that a message is waiting for a
// running generation (opencoti's POST /v1/notice, by the request's key).
const XollamaNoticePath = "/api/xollama/notice"

// FeatureGenerationNotice: the server passes a chat request's `notices` to an
// engine that plants generation notices, and serves XollamaNoticePath. Listed
// with GenerationNotice only while the engine this build runs declares it.
const FeatureGenerationNotice = "generation_notice_v1"

// NoticeKindMessageWaiting is the one kind of waiting call contract v1 has.
const NoticeKindMessageWaiting = "message_waiting"

// GenerationNotice is the capability block a client reads to opt in and to
// hide the planted lines: every line the engine plants starts with Marker and
// one space. The line stays in the content; the client hides it for display
// and keeps it in history, so the prompt cache holds.
type GenerationNotice struct {
	Version int    `json:"version"`
	Marker  string `json:"marker"`
}

// NoticeRequest is the body of a waiting call: the key the client put in the
// running request's `notices`, and the kind. No message text reaches the model.
type NoticeRequest struct {
	Key  string `json:"key"`
	Kind string `json:"kind"`
}
