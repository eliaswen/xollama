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

// FeatureGenerationNoticeV3: the engine also takes contract v3 (opencoti c13
// r5, mails #1576-#1582): a request's `notices` with "version": 3 gets plain
// sentences with no frame, and every planted line comes back as a Notice, on
// its own chunk before the text that holds it and in the done chunk's list.
// Listed beside FeatureGenerationNotice, never alone.
const FeatureGenerationNoticeV3 = "generation_notice_v3"

// Notice is one line the engine planted under contract v3, exactly as it
// reported it: Text is the line as streamed and Break the line break written
// after it, so a client hides the next occurrence of Text+Break in the
// message. Phase is "thinking", "answer" or "message"; Channel is where the
// engine saw it ("reasoning" inside the thinking part, else "content"). Cut
// marks a line the output cap ended, whose Break is what was written of it.
type Notice struct {
	Index   int    `json:"index"`
	Phase   string `json:"phase"`
	Channel string `json:"channel,omitempty"`
	Text    string `json:"text"`
	Break   string `json:"break"`
	Cut     bool   `json:"cut,omitempty"`
}

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
