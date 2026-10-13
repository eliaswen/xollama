package llm

import (
	"testing"

	"github.com/ollama/ollama/format"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/xollama"
)

// The two loads the estimate was written against, with the metadata of their
// GGUFs and the engine's own lines for what they took (eleven2go, 2026-10-11).

// omnimerge-v6-mtp 27B: a hybrid model, one block in four keeps a cache, and
// one NextN block for the drafter.
func qwen35KV() gguftest.KV {
	return gguftest.KV{
		"general.architecture":           "qwen35",
		"qwen35.block_count":             uint32(65),
		"qwen35.embedding_length":        uint32(5120),
		"qwen35.context_length":          uint32(262144),
		"qwen35.attention.head_count":    uint32(24),
		"qwen35.attention.head_count_kv": uint32(4),
		"qwen35.attention.key_length":    uint32(256),
		"qwen35.attention.value_length":  uint32(256),
		"qwen35.full_attention_interval": uint32(4),
		"qwen35.nextn_predict_layers":    uint32(1),
	}
}

// v9-agentic (Gemma 4 A4B): five blocks in six hold only the sliding window.
func gemma4KV() gguftest.KV {
	heads := make([]uint32, 30)
	pattern := make([]bool, 30)
	for i := range heads {
		heads[i], pattern[i] = 8, true
		if i%6 == 5 {
			heads[i], pattern[i] = 2, false
		}
	}
	return gguftest.KV{
		"general.architecture":                    "gemma4",
		"gemma4.block_count":                      uint32(30),
		"gemma4.embedding_length":                 uint32(2816),
		"gemma4.context_length":                   uint32(262144),
		"gemma4.attention.head_count":             uint32(16),
		"gemma4.attention.head_count_kv":          heads,
		"gemma4.attention.key_length":             uint32(512),
		"gemma4.attention.value_length":           uint32(512),
		"gemma4.attention.key_length_swa":         uint32(256),
		"gemma4.attention.value_length_swa":       uint32(256),
		"gemma4.attention.sliding_window":         uint32(1024),
		"gemma4.attention.sliding_window_pattern": pattern,
		"gemma4.attention.shared_kv_layers":       uint32(0),
	}
}

func mib(b uint64) uint64 { return (b + format.MebiByte/2) / format.MebiByte }

func q8() LlamaServerConfig {
	return LlamaServerConfig{Xollama: &xollama.Config{KV: &xollama.KV{K: "q8_0", V: "q8_0"}}}
}

// "KV demand 4250, 34.0 KiB/cell" and the draft context's
// "CUDA0 KV buffer size = 500.00 MiB" at 128000 cells. Upstream's formula
// gives 26 GiB here, which is what evicted a model for it.
func TestTheEstimateIsTheEnginesKVForAHybridModel(t *testing.T) {
	f := loadTestGGUF(t, qwen35KV())
	got := PredictOpencotiVRAM("", f, q8(), 128000, 128000, 512, 1, 0)
	if mib(got.KV) != 4250 {
		t.Errorf("KV = %d MiB, the engine sized 4250", mib(got.KV))
	}
	if mib(got.Drafter) != 500 {
		t.Errorf("drafter KV = %d MiB, the engine sized 500", mib(got.Drafter))
	}
	if got.SWA != 0 {
		t.Errorf("sliding-window KV = %d for a model with no window", got.SWA)
	}

	off := q8()
	off.DraftOff = true
	if got := PredictOpencotiVRAM("", f, off, 128000, 128000, 512, 1, 0); got.Drafter != 0 {
		t.Errorf("drafter KV = %d MiB with the drafter off", mib(got.Drafter))
	}
}

// "KV demand 996, 10.6 KiB/cell" for the blocks that cache the context and
// "KV demand 531" for the sliding window (5120 cells: four sequences of 1024
// and a 1024 batch), at 96000 cells.
func TestTheEstimateIsTheEnginesKVForASlidingWindowModel(t *testing.T) {
	t.Setenv("XOLLAMA_DYNAMIC_SLOTS", "true")
	t.Setenv("XOLLAMA_MAX_PARALLEL", "4")
	f := loadTestGGUF(t, gemma4KV())
	got := PredictOpencotiVRAM("", f, q8(), 96000, 96000, 1024, 1, 0)
	if mib(got.KV) != 996 {
		t.Errorf("KV = %d MiB, the engine sized 996", mib(got.KV))
	}
	if mib(got.SWA) != 531 {
		t.Errorf("sliding-window KV = %d MiB, the engine sized 531", mib(got.SWA))
	}
	if got.Drafter != 0 {
		t.Errorf("drafter KV = %d for a model with no NextN block", got.Drafter)
	}
}

// The cache is priced in the type the load runs with: the model's own, then
// the server's, then f16.
func TestTheEstimatePricesTheCacheType(t *testing.T) {
	f := loadTestGGUF(t, qwen35KV())
	kv := func(cfg LlamaServerConfig) uint64 {
		return mib(PredictOpencotiVRAM("", f, cfg, 128000, 128000, 512, 1, 0).KV)
	}
	t.Setenv("OLLAMA_KV_CACHE_TYPE", "")
	t.Setenv("XOLLAMA_K_CACHE_TYPE", "")
	t.Setenv("XOLLAMA_V_CACHE_TYPE", "")
	if got := kv(LlamaServerConfig{}); got != 8000 {
		t.Errorf("unstated type: KV = %d MiB, want f16's 8000", got)
	}
	t.Setenv("XOLLAMA_K_CACHE_TYPE", "q4_0")
	t.Setenv("XOLLAMA_V_CACHE_TYPE", "q4_0")
	if got := kv(LlamaServerConfig{}); got != 2250 {
		t.Errorf("server-wide q4_0: KV = %d MiB, want 2250", got)
	}
	if got := kv(q8()); got != 4250 {
		t.Errorf("the model's q8_0 over the server's q4_0: KV = %d MiB, want 4250", got)
	}
	kvarn := LlamaServerConfig{Xollama: &xollama.Config{KV: &xollama.KV{K: "kvarn3", V: "kvarn3"}}}
	if got := kv(kvarn); got != 1688 {
		t.Errorf("kvarn3 (3.375 bits): KV = %d MiB, want 1688", got)
	}
}

// The prediction leaves upstream's only when no backend the placement could
// pick ends on stock llama.cpp.
func TestTheOpencotiEstimateNeedsEveryBackendOnOpencoti(t *testing.T) {
	cuda := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, ComputeMajor: 8, ComputeMinor: 6}
	t.Setenv("XOLLAMA_ENGINE", "llamacpp")
	if EveryGroupOnOpencoti(LlamaServerConfig{}, []ml.DeviceInfo{cuda}) {
		t.Error("with XOLLAMA_ENGINE=llamacpp the estimate left upstream's")
	}
	if EveryGroupOnOpencoti(LlamaServerConfig{}, nil) {
		t.Error("a CPU load took the opencoti estimate")
	}
}

// The engine's "llama_memory_recurrent: size =" on the Mac mini, four
// sequences each: 38.53 MiB for Qwen3.5 2B, and 1197.00 MiB for the 27B with
// its built-in drafter at depth 3 (four rows a sequence).
func TestTheEstimateIsTheEnginesRecurrentState(t *testing.T) {
	small := gguftest.KV{
		"general.architecture":           "qwen35",
		"qwen35.block_count":             uint32(24),
		"qwen35.embedding_length":        uint32(2048),
		"qwen35.attention.head_count":    uint32(8),
		"qwen35.attention.head_count_kv": uint32(2),
		"qwen35.full_attention_interval": uint32(4),
		"qwen35.ssm.conv_kernel":         uint32(4),
		"qwen35.ssm.group_count":         uint32(16),
		"qwen35.ssm.inner_size":          uint32(2048),
		"qwen35.ssm.state_size":          uint32(128),
	}
	big := qwen35KV()
	big["qwen35.ssm.conv_kernel"] = uint32(4)
	big["qwen35.ssm.group_count"] = uint32(16)
	big["qwen35.ssm.inner_size"] = uint32(6144)
	big["qwen35.ssm.state_size"] = uint32(128)

	state := func(kv gguftest.KV, cfg LlamaServerConfig, depth int) float64 {
		return float64(PredictOpencotiVRAM("", loadTestGGUF(t, kv), cfg, 4096, 4096, 512, 1, depth).State) / format.MebiByte
	}
	rows := float64(PredictOpencotiVRAM("", loadTestGGUF(t, small), LlamaServerConfig{}, 4096, 4096, 512, 1, 0).State) / (18 * (18432 + 262144) * 2)
	if rows < 1 {
		t.Fatalf("no row for a sequence: %v", rows)
	}
	// Per sequence: 38.53 / 4 and 1197.00 / 4.
	if got, want := state(small, LlamaServerConfig{}, 3)/rows, 38.53/4; got < want-0.01 || got > want+0.01 {
		t.Errorf("Qwen3.5 2B: %.2f MiB a sequence, the engine takes %.2f (a model without a drafter has no rollback rows)", got, want)
	}
	if got, want := state(big, LlamaServerConfig{}, 3)/rows, 1197.0/4; got < want-0.01 || got > want+0.01 {
		t.Errorf("27B with its drafter at depth 3: %.2f MiB a sequence, the engine takes %.2f", got, want)
	}
	if got, want := state(big, LlamaServerConfig{DraftOff: true}, 3)/rows, 1197.0/16; got < want-0.01 || got > want+0.01 {
		t.Errorf("27B with the drafter off: %.2f MiB a sequence, want %.2f", got, want)
	}
	if got := state(gemma4KV(), LlamaServerConfig{}, 3); got != 0 {
		t.Errorf("a model with no recurrent block has %.2f MiB of state", got)
	}
}
