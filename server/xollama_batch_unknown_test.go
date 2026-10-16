package server

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

// pandorum, 2026-10-16: the card read "0 B" through a model swap and a 256k
// load took num_batch 2048, which cost the assistant drafter. Unknown memory
// keeps the default batch on opencoti; known room still earns the large one,
// and stock llama.cpp keeps upstream's answer.
func TestUnknownFreeMemoryKeepsTheDefaultBatchOnOpencoti(t *testing.T) {
	gpus := []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, TotalMemory: 16 << 30}}
	pick := func(available uint64) int {
		req := &LlmRequest{model: &Model{ModelPath: "model.gguf", ShortName: "m"}, opts: api.DefaultOptions(), numBatchAuto: true}
		req.applyAutomaticGenerationBatch(true, 256000, 12<<30, available, ml.FlashAttentionEnabled, gpus)
		return req.opts.NumBatch
	}

	t.Setenv("XOLLAMA_ENGINE", "opencoti")
	if got := pick(0); got != llamaServerGenerationBatchDefault {
		t.Errorf("opencoti, memory unknown: batch %d, want %d", got, llamaServerGenerationBatchDefault)
	}
	if got := pick(64 << 30); got != llamaServerGenerationBatchLarge {
		t.Errorf("opencoti, ample memory: batch %d, want %d", got, llamaServerGenerationBatchLarge)
	}

	t.Setenv("XOLLAMA_ENGINE", "llamacpp")
	if got := pick(0); got != llamaServerGenerationBatchLarge {
		t.Errorf("stock llama.cpp, memory unknown: batch %d, want upstream's %d", got, llamaServerGenerationBatchLarge)
	}
}
