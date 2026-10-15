package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"time"

	"github.com/ollama/ollama/api"
)

// drafterStateFeature is the /props feature under which opencoti keeps a
// record of the drafter each load ended with (c11 r3, patch 0618).
const drafterStateFeature = "drafter_state_v1"

// drafterStateTimeout bounds the read: `ps` and `show` wait on it, and the
// engine answers it from a record, never from the context.
const drafterStateTimeout = 2 * time.Second

// DrafterState is the engine's record of the drafter this load ended with, or
// nil where the engine keeps none: stock llama.cpp, an opencoti without
// drafter_state_v1, or an engine that does not answer. The configuration and
// `show` name the drafter the model carries; only the engine knows whether it
// dropped it at load to keep the KV window resident.
func (s *llamaServerRunner) DrafterState(ctx context.Context) *api.DrafterState {
	if !s.usedOpencoti || s.port == 0 {
		return nil
	}
	ctx, cancel := context.WithTimeout(ctx, drafterStateTimeout)
	defer cancel()
	status, out, err := s.engineRequest(ctx, http.MethodGet, "/props", nil)
	if err != nil || status != http.StatusOK {
		return nil
	}
	return parseDrafterState(out)
}

// parseDrafterState reads opencoti.speculative.drafter from a /props body. A
// state newer than this build is shown as dropped: whatever it is, it is not
// the attached drafter the configuration names (agreed with opencoti, #960).
func parseDrafterState(props []byte) *api.DrafterState {
	var p struct {
		Features []string `json:"features"`
		Opencoti struct {
			Speculative struct {
				Drafter *api.DrafterState `json:"drafter"`
			} `json:"speculative"`
		} `json:"opencoti"`
	}
	if json.Unmarshal(props, &p) != nil || !slices.Contains(p.Features, drafterStateFeature) {
		return nil
	}
	d := p.Opencoti.Speculative.Drafter
	if d == nil {
		return nil
	}
	switch d.State {
	case "attached", "dropped", "none":
	default:
		d.State = "dropped"
	}
	return d
}
