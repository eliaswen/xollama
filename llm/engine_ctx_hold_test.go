package llm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"

	"github.com/ollama/ollama/llm/engine"
)

func holdEngine(t *testing.T, declares bool) *llamaServerRunner {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f := []string{"polykv_v1"}
		if declares {
			f = append(f, engine.FeatureWindowUnheld)
		}
		json.NewEncoder(w).Encode(map[string]any{"features": f})
	}))
	t.Cleanup(srv.Close)
	u, _ := url.Parse(srv.URL)
	port, _ := strconv.Atoi(u.Port())
	return &llamaServerRunner{port: port, client: srv.Client(), usedOpencoti: true}
}

// An unbooked window (owner, 2026-10-15): the client's num_ctx_hold goes to
// the engine beside the window it qualifies, and only to an engine that
// declares elastic_window_unheld_v1; every other body is unchanged.
func TestNumCtxHoldGoesOnlyBesideAWindowToAnEngineThatTakesIt(t *testing.T) {
	no := false
	p := &Placement{NumCtx: 65536, NumCtxMin: 16384, NumCtxHold: &no}

	if got := holdEngine(t, true).ctxHoldFor(t.Context(), p, 65536); got == nil || *got {
		t.Fatalf("an engine with the feature got %v, want false", got)
	}
	if got := holdEngine(t, false).ctxHoldFor(t.Context(), p, 65536); got != nil {
		t.Fatalf("an engine without the feature got %v", *got)
	}
	if got := holdEngine(t, true).ctxHoldFor(t.Context(), p, 0); got != nil {
		t.Fatal("num_ctx_hold sent with no window")
	}
	if got := holdEngine(t, true).ctxHoldFor(t.Context(), &Placement{NumCtx: 65536}, 65536); got != nil {
		t.Fatal("num_ctx_hold sent although the client stated none")
	}
	if got := (&llamaServerRunner{}).ctxHoldFor(t.Context(), p, 65536); got != nil {
		t.Fatal("stock llama.cpp got num_ctx_hold")
	}
}
