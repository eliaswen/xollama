package llm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sync"
	"time"

	"github.com/ollama/ollama/llm/engine"
	"github.com/ollama/ollama/ml"
)

// xollama-hook: notices
//
// Generation notices, opencoti contract v2 (c13 r3; mails #1487-#1495): a
// client opts in per request with a top-level `notices` object, and the engine
// plants a templated notice inline in the model's own output when a phase's
// room runs low or a message is waiting. xollama passes the object through
// unchanged, leaves the notice in the content (the client hides it and keeps
// it in history, owner 2026-10-14), starts the engine with the model's
// templates (appendNoticeTemplatesArgs), and routes the waiting call to the
// runner holding the request with that key.

// noticeTimeout bounds a waiting call: the engine answers it from a table.
const noticeTimeout = 5 * time.Second

// ErrNoNoticeKey is a waiting call whose key no running request carries.
var ErrNoNoticeKey = errors.New("no running request has this notice key")

var noticeKeys = struct {
	sync.Mutex
	m map[string]*llamaServerRunner
}{m: map[string]*llamaServerRunner{}}

// noticesFor is what this request sends the engine as `notices`: the client's
// object unchanged on an engine that declares generation_notice_v2, nothing on
// any other, so a stock llama.cpp body stays upstream's.
func (s *llamaServerRunner) noticesFor(ctx context.Context, raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 || !s.usedOpencoti || !s.Features(ctx)[featureGenerationNotice] {
		return nil
	}
	return raw
}

const featureGenerationNotice = engine.FeatureGenerationNotice

// appendNoticeTemplatesArgs starts the engine with the model's own notice
// templates layer, loaded from the store as it is. A model without one
// passes nothing: the engine's built-in ladder is xollama's default. Only on
// opencoti, and only on an engine that takes the flag: an older one refuses
// the whole command line for it.
func appendNoticeTemplatesArgs(args []string, path string, usedOpencoti bool) []string {
	if path == "" || !usedOpencoti || !engineTakesNoticeTemplates() {
		return args
	}
	return append(args, "--notice-templates", path)
}

// engineTakesNoticeTemplates is engine.NoticesServed behind a seam, so a test
// does not run the engine.
var engineTakesNoticeTemplates = func() bool { return engine.NoticesServed(ml.LibOllamaPath) }

// trackNotice keeps the request's key routable for the waiting call while the
// request runs, and returns the release. A key already held by another
// running request stays with it: the engine answers 409 for the second.
func (s *llamaServerRunner) trackNotice(notices json.RawMessage) func() {
	var o struct {
		Key string `json:"key"`
	}
	if json.Unmarshal(notices, &o) != nil || o.Key == "" {
		return func() {}
	}
	noticeKeys.Lock()
	defer noticeKeys.Unlock()
	if _, held := noticeKeys.m[o.Key]; held {
		return func() {}
	}
	noticeKeys.m[o.Key] = s
	return func() {
		noticeKeys.Lock()
		defer noticeKeys.Unlock()
		if noticeKeys.m[o.Key] == s {
			delete(noticeKeys.m, o.Key)
		}
	}
}

// PostNotice sends a waiting call to the engine running the request with that
// key, and returns the engine's own status and body (200, 400, 404, 409).
func PostNotice(ctx context.Context, key, kind string) (int, []byte, error) {
	noticeKeys.Lock()
	s := noticeKeys.m[key]
	noticeKeys.Unlock()
	if s == nil {
		return http.StatusNotFound, nil, ErrNoNoticeKey
	}
	body, err := json.Marshal(map[string]string{"key": key, "kind": kind})
	if err != nil {
		return 0, nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, noticeTimeout)
	defer cancel()
	return s.engineRequest(ctx, http.MethodPost, "/v1/notice", body)
}
