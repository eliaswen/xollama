package server

import (
	"context"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

// xollama-hook: drafter
//
// The engine can drop a configured drafter at load, when the head would keep
// the KV window from being resident (opencoti 0612, the bug-3740 relief). The
// model's configuration still names the drafter, so `show` listed it as
// attached and `ps` gave no sign, while the only trace was a log line.
// opencoti keeps a record of what each load ended with (drafter_state_v1,
// c11 r3); these read it for `ps` and `show`. Stock llama.cpp keeps none, and
// both answers stay upstream's there.

// drafterReporter is a runner whose engine keeps a drafter record.
type drafterReporter interface {
	DrafterState(context.Context) *api.DrafterState
}

func loadedDrafterState(ctx context.Context, llama llm.LlamaServer) *api.DrafterState {
	r, ok := llama.(drafterReporter)
	if !ok || llama == nil {
		return nil
	}
	return r.DrafterState(ctx)
}

// psDrafter is the drafter record `ps` lists: only a dropped one, the case a
// listing would otherwise get wrong.
func psDrafter(ctx context.Context, llama llm.LlamaServer) *api.DrafterState {
	if d := loadedDrafterState(ctx, llama); d != nil && d.State == "dropped" {
		return d
	}
	return nil
}

// withLoadedDrafter adds to a show response what the engine serving the model
// now did with its drafter. resp may be the show cache's, so it is copied,
// never changed.
func (s *Server) withLoadedDrafter(ctx context.Context, name model.Name, resp *api.ShowResponse) *api.ShowResponse {
	if resp == nil || resp.Drafter == nil || resp.Drafter.Source == api.DrafterSourceNone || s.sched == nil {
		return resp
	}
	for _, lm := range s.sched.loadedModels() {
		if lm.model == nil || lm.llama == nil || !model.ParseName(lm.model.Name).EqualFold(name) {
			continue
		}
		state := loadedDrafterState(ctx, lm.llama)
		if state == nil {
			return resp
		}
		d := *resp.Drafter
		d.Loaded = state
		out := *resp
		out.Drafter = &d
		return &out
	}
	return resp
}
