package server

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm/engine"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/xollama"
)

// xollama-hook: notices — see docs/xollama/notices.mdx.
//
// A model's notice templates are a layer of their own (xollama.MediaTypeNotices)
// whose blob is passed to the engine as --notice-templates. A model without
// one starts with xollama's defaults, so the server always knows what is
// planted and which spans a client hides (owner, 2026-10-15).

// noticeConfigCheck has the engine judge a templates file: nil, or its own
// one-line reason. A variable so a route test does not run the engine.
var noticeConfigCheck = func(ctx context.Context, file string) error {
	_, err := engine.PrintNoticeConfig(ctx, ml.LibOllamaPath, file)
	return err
}

// applyNoticeTemplates makes the model's notices layer what the request's
// notices.templates says: a digest attaches that blob once the engine has
// accepted it, xollama.NoticeTemplatesNone detaches the model's. A request
// that says nothing leaves the layer, inherited through FROM or not, as it
// is. The block itself is never stored: the layer is the truth.
//
// xollama-hook: notices (called from CreateHandler)
func applyNoticeTemplates(ctx context.Context, r *api.CreateRequest, layers []*modelLayer) ([]*modelLayer, error) {
	if r.Xollama == nil || r.Xollama.Notices.IsZero() {
		return layers, nil
	}
	digest := r.Xollama.Notices.Templates
	r.Xollama.Notices = nil

	kept := layers[:0:0]
	for _, l := range layers {
		if l.MediaType != xollama.MediaTypeNotices {
			kept = append(kept, l)
		}
	}
	if digest == xollama.NoticeTemplatesNone {
		return kept, nil
	}

	blob, err := manifest.BlobsPath(digest)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(blob)
	if err != nil {
		return nil, fmt.Errorf("notices.templates %s is not in the model store; upload it first: %w", digest, err)
	}
	// A part the file leaves out is xollama's default, not the engine's
	// built-in: the spans a client hides must cover everything planted.
	complete, err := xollama.CompleteNoticeTemplates(data)
	if err != nil {
		return nil, err
	}
	if !noticesServed() {
		return nil, fmt.Errorf("notices.templates: the engine this server runs does not plant generation notices (%s), so it cannot check the templates", engine.FeatureGenerationNotice)
	}
	layer, err := manifest.NewLayer(bytes.NewReader(complete), xollama.MediaTypeNotices)
	if err != nil {
		return nil, err
	}
	if blob, err = manifest.BlobsPath(layer.Digest); err != nil {
		return nil, err
	}
	if err := noticeConfigCheck(ctx, blob); err != nil {
		return nil, fmt.Errorf("notices.templates: the engine refused the file: %w", err)
	}
	return append(kept, &modelLayer{Layer: layer}), nil
}

// noticeTemplatesPath is the model's own templates layer, passed to the
// engine as it is, or "" when it has none or the server does not serve
// notices. A model without a layer gets no --notice-templates: the engine
// plants its built-in ladder, which is xollama.DefaultNoticeTemplates (held
// equal at every pin move), so nothing is ever written for it.
func noticeTemplatesPath(m *Model) string {
	if m == nil || !noticesServed() {
		return ""
	}
	return m.NoticeTemplatesPath
}

// modelGenerationNotice is /api/show's block: the templates the model starts
// with and the spans a client hides. Nil while notices are not served.
func modelGenerationNotice(m *Model) *api.GenerationNotice {
	if m == nil || !noticesServed() {
		return nil
	}
	data := xollama.DefaultNoticeTemplates
	if m.NoticeTemplatesPath != "" {
		b, err := os.ReadFile(m.NoticeTemplatesPath)
		if err != nil {
			slog.Warn("notices: could not read the model's templates", "model", m.ShortName, "error", err)
			return nil
		}
		data = b
	}
	return noticeBlock(data)
}

// noticeBlock is the GenerationNotice of one templates file.
func noticeBlock(data []byte) *api.GenerationNotice {
	t, err := xollama.ParseNoticeTemplates(data)
	if err != nil {
		slog.Warn("notices: unreadable templates", "error", err)
		return nil
	}
	return &api.GenerationNotice{Version: 2, Hide: t.Hide(), Templates: bytes.TrimSpace(data)}
}
