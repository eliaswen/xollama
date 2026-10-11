package server

import (
	"log/slog"
	"slices"
	"sort"
	"strings"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// Eviction is the last resort on opencoti (the evict-last hook).
//
// Upstream predicts a load as its file plus an f16 KV cache on every block at
// the whole context, and evicts a loaded model when that exceeds 80% of the
// free memory. On eleven2go (RTX 3090 + RX 9070 XT, 2026-10-11) that was
// 42.7 GiB for a load that takes 21.5 and 16.2 GiB for one that takes 14.5:
// with one model on the 3090 and the 9070 XT empty, two clients evicted each
// other's model on every request, a minute a turn.
//
// Two things change where opencoti serves the load, and nothing where stock
// llama.cpp does (XOLLAMA_ENGINE=llamacpp is upstream's, to the byte):
//
//   - the prediction is the engine's own arithmetic on the GGUF
//     (llm.PredictOpencotiVRAM), so the placement picks a card on a real number;
//   - a load evicts only when its weights do not fit: the KV cache never
//     evicts, because the engine keeps what the card cannot hold in host
//     memory (the rolling window). Weights may go to host memory too, up to
//     maxWeightsOffCardPercent of them; past that the load is slow enough that
//     unloading an idle model is the better deal.

// maxWeightsOffCardPercent is how much of a model's weights may be left to
// host memory before a load evicts instead (owner, 2026-10-11).
const maxWeightsOffCardPercent = 20

// predictVRAM is the memory this load is placed on: opencoti's own figure
// when opencoti serves it on whichever backend the placement picks, else
// upstream's with the slot ceiling.
func predictVRAM(req *LlmRequest, f *gguf.Model, gpus []ml.DeviceInfo, predictedCtx, numParallel int) uint64 {
	cfg := llamaServerConfigForModel(req.model)
	if !llm.EveryGroupOnOpencoti(cfg, gpus) {
		return llm.PredictServerVRAM(req.model.ModelPath, f, predictedCtx) + slotCeilingVRAM(req, f, gpus, numParallel)
	}
	numBatch := req.opts.NumBatch
	if numBatch <= 0 {
		numBatch = llamaServerGenerationBatchDefault
	}
	return llm.PredictOpencotiVRAM(req.model.ModelPath, f, cfg, predictedCtx,
		effectiveModelContext(req.opts.NumCtx, f, req.contextUnlocked(f, gpus)), numBatch, numParallel).Total()
}

// needsEviction decides whether a load that has to share its GPUs evicts.
// predicted and free are what upstream compares, and for stock llama.cpp the
// answer is upstream's.
func needsEviction(req *LlmRequest, f *gguf.Model, loadGpus []ml.DeviceInfo, predicted, free uint64) bool {
	if !llm.WouldUseOpencoti(llamaServerConfigForModel(req.model), loadGpus) {
		return predicted > free*80/100
	}
	weights := llm.PredictOpencotiVRAM(req.model.ModelPath, f, llamaServerConfigForModel(req.model), 0, 0, 0, 1).Weights
	onCard := weights * (100 - maxWeightsOffCardPercent) / 100
	if free >= onCard {
		if predicted > free {
			slog.Info("the load does not fit this GPU whole; loading without evicting, the engine keeps the rest in host memory",
				"model", req.model.ShortName,
				"predicted", format.HumanBytes2(predicted),
				"weights", format.HumanBytes2(weights),
				"available", format.HumanBytes2(free))
		}
		return false
	}
	slog.Info("more than a fifth of the model's weights would not fit this GPU",
		"model", req.model.ShortName,
		"weights", format.HumanBytes2(weights),
		"available", format.HumanBytes2(free))
	return true
}

// gpuKey names a physical GPU: its PCI ID when discovery has one, so the CUDA
// and Vulkan copies of one card are one GPU, else its backend and index.
func gpuKey(g ml.DeviceInfo) string {
	if g.PCIID != "" {
		return strings.ToLower(g.PCIID)
	}
	return g.Library + "/" + g.ID
}

// busyGPUs is the runners on each GPU. It is read from the whole device list,
// before a pin or the GPU policy narrows it, so a runner on a card's other
// backend still counts.
func (s *Scheduler) busyGPUs(all []ml.DeviceInfo) map[string][]*runnerRef {
	s.loadedMu.Lock()
	runners := make([]*runnerRef, 0, len(s.loaded))
	for _, r := range s.loaded {
		runners = append(runners, r)
	}
	s.loadedMu.Unlock()

	busy := map[string][]*runnerRef{}
	for _, r := range runners {
		r.refMu.Lock()
		for _, g := range all {
			if k := gpuKey(g); runnerOn(r, g) && !slices.Contains(busy[k], r) {
				busy[k] = append(busy[k], r)
			}
		}
		r.refMu.Unlock()
	}
	return busy
}

// runnerOn reports whether a runner holds this device. refMu must be held.
func runnerOn(r *runnerRef, g ml.DeviceInfo) bool {
	for _, id := range r.gpus {
		if id == g.DeviceID {
			return true
		}
	}
	return r.llama != nil && r.llama.VRAMByGPU(g.DeviceID) > 0
}

// findRunnerToUnloadOn picks the runner to evict for a load that goes to these
// GPUs. Upstream evicts the runner idle the longest wherever it is; a runner on
// another card frees nothing the load can use, so one on these cards goes
// first. With none there, or a load stock llama.cpp serves, upstream decides.
func (s *Scheduler) findRunnerToUnloadOn(req *LlmRequest, gpus []ml.DeviceInfo, busy map[string][]*runnerRef) *runnerRef {
	if req.model == nil || len(gpus) == 0 || !llm.WouldUseOpencoti(llamaServerConfigForModel(req.model), gpus) {
		return s.findRunnerToUnload()
	}
	var on []*runnerRef
	s.loadedMu.Lock()
	for _, g := range gpus {
		for _, r := range busy[gpuKey(g)] {
			// busy was read before the load was tried: a runner that
			// has gone since is not one to wait for.
			if s.loaded[r.modelKey] == r && !slices.Contains(on, r) {
				on = append(on, r)
			}
		}
	}
	s.loadedMu.Unlock()
	if len(on) == 0 {
		return s.findRunnerToUnload()
	}
	sort.Sort(ByDurationAndName(on))
	for _, r := range on {
		r.refMu.Lock()
		rc := r.refCount
		r.refMu.Unlock()
		if rc == 0 {
			slog.Debug("found an idle runner on the load's GPUs to unload", "runner", r)
			return r
		}
	}
	slog.Debug("no idle runner on the load's GPUs, picking the shortest duration", "runner", on[0])
	return on[0]
}
