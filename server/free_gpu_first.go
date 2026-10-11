package server

import (
	"log/slog"
	"os"
	"slices"
	"sort"
	"strings"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// freeGPUFloorPercent is how much of the model file a free GPU has to hold
// before a load takes it instead of evicting: below that most of the model
// would run from host memory, and unloading an idle model is the better deal.
const freeGPUFloorPercent = 75

// A load that upstream's prediction says does not fit takes a GPU no runner
// is on before it evicts anything (the free-gpu-first hook).
//
// Upstream predicts a load as its file plus an f16 KV cache at the whole
// context and evicts when that exceeds 80% of the free memory. opencoti does
// not load that way: it fits itself to the card it is given, with the KV cache
// in the type the model states and its tail in host memory when the card is
// small. On eleven2go (RTX 3090 + RX 9070 XT, 2026-10-11) the prediction was
// 42.7 GiB for a model that takes 21.5, and 16.2 GiB for one that takes 14.5
// and had run on the 9070 XT that morning; with one model on the 3090 and the
// 9070 XT empty, two clients evicted each other's model on every request, a
// minute a turn.
//
// Stock llama.cpp keeps upstream's decision: nothing here answers for a GPU
// opencoti would not serve, so XOLLAMA_ENGINE=llamacpp changes nothing.

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

// freeGPUFirst returns the GPU a load takes instead of evicting, or nil when
// eviction is the answer: no discrete GPU opencoti serves is free of runners,
// or the free one is too small for most of the model.
func freeGPUFirst(req *LlmRequest, gpus []ml.DeviceInfo, busy map[string][]*runnerRef) []ml.DeviceInfo {
	if req.model == nil || req.model.IsMLX() || isMediaKey(req.model.ModelPath) ||
		req.opts.NumGPU == 0 || req.opts.MainGPU != nil {
		return nil
	}
	size := modelBytes(req.model)
	if size == 0 {
		return nil
	}
	cfg := llamaServerConfigForModel(req.model)
	var best *ml.DeviceInfo
	for i := range gpus {
		g := &gpus[i]
		if g.Integrated || len(busy[gpuKey(*g)]) > 0 || g.FreeMemory < size*freeGPUFloorPercent/100 ||
			!llm.WouldUseOpencoti(cfg, []ml.DeviceInfo{*g}) {
			continue
		}
		if best == nil || betterPlacementGPU(*g, g.FreeMemory, *best, best.FreeMemory) {
			best = g
		}
	}
	if best == nil {
		return nil
	}
	slog.Info("loading on a free GPU instead of evicting a loaded model",
		"model", req.model.ShortName,
		"library", best.Library,
		"id", best.ID,
		"description", best.Description,
		"available", format.HumanBytes2(best.FreeMemory),
		"model_size", format.HumanBytes2(size))
	return []ml.DeviceInfo{*best}
}

// modelBytes is the size of a model's GGUF, shards included.
func modelBytes(m *Model) uint64 {
	var total uint64
	for _, p := range append([]string{m.ModelPath}, m.ModelShardPaths...) {
		if fi, err := os.Stat(p); err == nil && fi.Size() > 0 {
			total += uint64(fi.Size())
		}
	}
	return total
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
