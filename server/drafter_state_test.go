package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

type drafterLlm struct {
	*mockLlm
	state *api.DrafterState
}

func (d *drafterLlm) DrafterState(context.Context) *api.DrafterState { return d.state }

func mib(n int64) *int64 { return &n }

func TestPsListsOnlyADrafterTheEngineDropped(t *testing.T) {
	dropped := &api.DrafterState{State: "dropped", Kind: "assistant", Reason: "fit_shortfall", ShortfallMiB: mib(51), FreedMiB: mib(614)}
	if got := psDrafter(t.Context(), &drafterLlm{mockLlm: &mockLlm{}, state: dropped}); got != dropped {
		t.Fatalf("a dropped drafter was not listed: %+v", got)
	}
	for _, state := range []string{"attached", "none"} {
		if got := psDrafter(t.Context(), &drafterLlm{mockLlm: &mockLlm{}, state: &api.DrafterState{State: state}}); got != nil {
			t.Fatalf("%s: listed %+v, want nothing", state, got)
		}
	}
	// Stock llama.cpp keeps no record: the listing stays upstream's.
	if got := psDrafter(t.Context(), &mockLlm{}); got != nil {
		t.Fatalf("a runner without a record listed %+v", got)
	}
	if got := psDrafter(t.Context(), nil); got != nil {
		t.Fatalf("a loading model listed %+v", got)
	}
}

func TestShowSaysWhatTheLoadedEngineDidWithTheDrafter(t *testing.T) {
	dropped := &api.DrafterState{State: "dropped", Kind: "assistant", Reason: "fit_shortfall"}
	s := &Server{sched: &Scheduler{loaded: map[string]*runnerRef{
		"a": engineRunner("gemma4:26b", &drafterLlm{mockLlm: &mockLlm{}, state: dropped}),
	}}}
	cached := &api.ShowResponse{Drafter: &api.DrafterInfo{Source: "attached", SpecType: "draft-mtp"}}

	got := s.withLoadedDrafter(t.Context(), model.ParseName("gemma4:26b"), cached)
	if got.Drafter.Loaded != dropped {
		t.Fatalf("show of a loaded model: loaded = %+v, want the engine's record", got.Drafter.Loaded)
	}
	if cached.Drafter.Loaded != nil {
		t.Fatal("the show cache's response was changed")
	}

	if other := s.withLoadedDrafter(t.Context(), model.ParseName("qwen3:8b"), cached); other != cached {
		t.Fatal("a model that is not loaded got a record")
	}
	none := &api.ShowResponse{Drafter: &api.DrafterInfo{Source: api.DrafterSourceNone}}
	if got := s.withLoadedDrafter(t.Context(), model.ParseName("gemma4:26b"), none); got != none {
		t.Fatal("a model with no drafter got a record")
	}
}

// The hook line in PsHandler: without it the record never reaches a client.
func TestPsHandlerCarriesTheDroppedDrafter(t *testing.T) {
	gin.SetMode(gin.TestMode)
	dropped := &api.DrafterState{State: "dropped", Kind: "assistant", Reason: "fit_shortfall"}
	s := &Server{sched: &Scheduler{loaded: map[string]*runnerRef{
		"a": engineRunner("gemma4:26b", &drafterLlm{mockLlm: &mockLlm{}, state: dropped}),
		"b": engineRunner("qwen3:8b", &mockLlm{}),
	}}}
	r := gin.New()
	r.GET("/api/ps", s.PsHandler)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/ps", nil))

	var resp api.ProcessResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	got := map[string]*api.DrafterState{}
	for _, m := range resp.Models {
		got[m.Name] = m.Drafter
	}
	if d := got["gemma4:26b"]; d == nil || d.State != "dropped" || d.Reason != "fit_shortfall" {
		t.Fatalf("gemma4:26b drafter = %+v, want the dropped record", d)
	}
	if d := got["qwen3:8b"]; d != nil {
		t.Fatalf("qwen3:8b drafter = %+v, want none", d)
	}
}
