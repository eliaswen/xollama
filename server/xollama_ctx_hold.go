package server

import (
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm/engine"
)

// xollama-hook: context-window-unbooked — Placement.NumCtxHold
// (llm/engine_ctx_hold.go).

// unbookedServed reports whether a client's Placement.NumCtxHold reaches the
// engine: the pinned engine declares elastic_window_unheld_v1 and the server
// is not switched to stock llama.cpp. A variable so a route test can set it.
var unbookedServed = func() bool {
	return !strings.EqualFold(envconfig.Engine(), "llamacpp") && engine.HasWindowUnheld()
}
