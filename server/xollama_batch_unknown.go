package server

import (
	"log/slog"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// xollama-hook: batch-unknown-memory
//
// Upstream's automatic batch takes an unknown free memory (0) as room for
// any batch, so a load placed while the refresh could not read the card got
// the largest one. On pandorum (2026-10-16, RTX 5080, a model swapped from
// 128k to 256k) the refresh failed through the swap, the card read "0 B", and
// the load took num_batch 2048 instead of 512: a 1410 MiB compute buffer,
// and opencoti dropped the assistant drafter for 299 MiB (decode 98-116 tok/s
// against 145-198 with it). On opencoti, memory nobody could read keeps the
// default batch; the engine's fit sizes the rest. Stock llama.cpp keeps
// upstream's answer exactly.
func batchForUnknownMemory(req *LlmRequest, batch int, availableMemory uint64, gpus []ml.DeviceInfo) int {
	if availableMemory != 0 || batch <= llamaServerGenerationBatchDefault ||
		!llm.WouldUseOpencoti(llamaServerConfigForModel(req.model), gpus) {
		return batch
	}
	slog.Info("free memory unknown at placement; keeping the default batch", "model", req.model.ShortName, "batch", llamaServerGenerationBatchDefault, "would_have", batch)
	return llamaServerGenerationBatchDefault
}
