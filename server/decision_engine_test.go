package server

import (
	"testing"

	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/types/xollama"
)

func decisionModel(t *testing.T, decisionType string) *Model {
	t.Helper()
	kv := gguftest.KV{"general.architecture": "clef"}
	if decisionType != "" {
		kv["clef.decision.type"] = decisionType
	}
	path, _ := createBinFile(t, kv, []*gguftest.Tensor{})
	m := &Model{ModelPath: path, ShortName: "m"}
	loadTestMetadata(t, m)
	return m
}

func withSystemOne(t *testing.T, has bool) {
	t.Helper()
	old := engineHasSystemOne
	engineHasSystemOne = func() bool { return has }
	t.Cleanup(func() { engineHasSystemOne = old })
}

func TestADecisionModelGoesToStockWhileTheEngineHasNoSystemOne(t *testing.T) {
	withSystemOne(t, false)
	for _, kind := range []string{"clef", "nimble", "systemone"} {
		m := decisionModel(t, kind)

		if got := decisionEngine(m, nil); got == nil || got.Engine != xollama.EngineLlamaCpp {
			t.Fatalf("%s with no settings: engine = %+v, want %q", kind, got, xollama.EngineLlamaCpp)
		}

		own := &xollama.Config{FlashAttention: "on"}
		got := decisionEngine(m, own)
		if got.Engine != xollama.EngineLlamaCpp || got.FlashAttention != "on" {
			t.Fatalf("%s with settings: %+v, want its settings kept and engine %q", kind, got, xollama.EngineLlamaCpp)
		}
		if own.Engine != "" {
			t.Fatalf("%s: the model's own config was changed: engine = %q", kind, own.Engine)
		}
	}
}

func TestOnlyADecisionModelWithoutItsOwnEngineIsMoved(t *testing.T) {
	withSystemOne(t, false)

	stated := &xollama.Config{Engine: xollama.EngineOpencoti}
	if got := decisionEngine(decisionModel(t, "clef"), stated); got != stated {
		t.Fatalf("an engine the model states was replaced: %+v", got)
	}
	if got := decisionEngine(decisionModel(t, ""), nil); got != nil {
		t.Fatalf("a chat model was moved: %+v", got)
	}
}

func TestADecisionModelStaysOnAnEngineThatServesSystemOne(t *testing.T) {
	withSystemOne(t, true)
	if got := decisionEngine(decisionModel(t, "clef"), nil); got != nil {
		t.Fatalf("the engine declares systemone_v1, yet the model was moved: %+v", got)
	}
}

// The launch reads the engine from launchXollama; a decisionEngine nobody calls
// would pass every test above.
func TestTheLaunchConfigOfADecisionModelNamesStock(t *testing.T) {
	withSystemOne(t, false)
	if got := launchXollama(decisionModel(t, "nimble")); got == nil || got.Engine != xollama.EngineLlamaCpp {
		t.Fatalf("launch config of a decision model: %+v, want engine %q", got, xollama.EngineLlamaCpp)
	}
}
