package llm

import (
	"context"

	"github.com/ollama/ollama/llm/engine"
)

// xollama-hook: context-window-unbooked
//
// A client's unbooked window (owner ruling 2026-10-15, opencoti #1504/#1511):
// a stated num_ctx gets that size if it is free, or less down to num_ctx_min,
// and is not held for the session after the request. xollama always sends a
// session_id for slot and prompt-cache affinity, which the engine would take
// as a booking, so the client says so with Placement.NumCtxHold and the engine
// gets num_ctx_hold.

// ctxHoldFor is the num_ctx_hold a placed request carries: the client's, beside
// a window, on an engine whose /props declares elastic_window_unheld_v1. Nil
// otherwise, so every other body is unchanged.
func (s *llamaServerRunner) ctxHoldFor(ctx context.Context, p *Placement, numCtx int) *bool {
	if p == nil || p.NumCtxHold == nil || numCtx <= 0 || !s.usedOpencoti || !s.Features(ctx)[engine.FeatureWindowUnheld] {
		return nil
	}
	return p.NumCtxHold
}
