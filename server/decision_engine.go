package server

import (
	"log/slog"
	"sync"

	"github.com/ollama/ollama/llm/engine"
	"github.com/ollama/ollama/types/xollama"
)

// engineHasSystemOne is what the pinned engine declares; a variable so a test
// can state either answer.
var engineHasSystemOne = engine.HasSystemOne

// decisionSaid holds the models already named in the log: the launch config is
// built for every request, and the line is worth reading once.
var decisionSaid sync.Map

// decisionEngine sends a decision model to stock llama-server while the pinned
// opencoti engine does not serve /v1/systemone. Since v0.40.4 upstream scores
// every GGUF decision model (Clef, Nimble and the rest) with one request to
// that route, llama.cpp b11521's native decision support; an engine without it
// cannot answer the call (c12 r2 has no such route, 2026-10-14). An engine the
// model states itself is kept, and once the pin declares systemone_v1 this
// returns cfg untouched. Where no stock llama-server ships (macOS), the load is
// refused by name (engine.StocklessRefusal).
func decisionEngine(m *Model, cfg *xollama.Config) *xollama.Config {
	if m == nil || m.metadata.String("decision.type") == "" || engineHasSystemOne() {
		return cfg
	}
	if cfg != nil && cfg.Engine != "" {
		return cfg
	}
	out := xollama.Config{}
	if cfg != nil {
		out = *cfg
	}
	out.Engine = xollama.EngineLlamaCpp
	if _, said := decisionSaid.LoadOrStore(m.ShortName, true); !said {
		slog.Info("decision model is served by stock llama-server: the pinned opencoti engine does not serve /v1/systemone",
			"model", m.ShortName, "decision_type", m.metadata.String("decision.type"))
	}
	return &out
}
