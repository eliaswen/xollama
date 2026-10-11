package server

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// eleven2go's two cards as the scheduler sees them, the 3090 with a model on it.
func elevenCards() (cuda, vulkan ml.DeviceInfo) {
	cuda = ml.DeviceInfo{
		DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, PCIID: "0000:01:00.0", Description: "NVIDIA GeForce RTX 3090",
		ComputeMajor: 8, ComputeMinor: 6, TotalMemory: 24 << 30, FreeMemory: 10 << 30,
	}
	vulkan = ml.DeviceInfo{
		DeviceID: ml.DeviceID{ID: "2", Library: "Vulkan"}, PCIID: "0000:03:00.0", Description: "AMD Radeon RX 9070 XT",
		TotalMemory: 16 << 30, FreeMemory: 15 << 30,
	}
	return cuda, vulkan
}

func skipUnlessOpencotiServesBoth(t *testing.T, cuda, vulkan ml.DeviceInfo) {
	t.Helper()
	t.Setenv("XOLLAMA_ENGINE", "")
	if !llm.WouldUseOpencoti(llm.LlamaServerConfig{}, []ml.DeviceInfo{cuda}) || !llm.WouldUseOpencoti(llm.LlamaServerConfig{}, []ml.DeviceInfo{vulkan}) {
		t.Skip("this build's pin does not serve both CUDA and Vulkan here; the case does not arise")
	}
}

func loadedOn(s *Scheduler, name string, idle time.Duration, on ...ml.DeviceInfo) *runnerRef {
	r := &runnerRef{
		llama:           &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}},
		modelKey:        name,
		sessionDuration: idle,
	}
	for _, g := range on {
		r.gpus = append(r.gpus, g.DeviceID)
	}
	s.loadedMu.Lock()
	s.loaded[name] = r
	s.loadedMu.Unlock()
	return r
}

// A load upstream's prediction says does not fit takes the card no runner is on
// instead of evicting the model on the other card: on eleven2go two clients
// evicted each other's model from the 3090 on every request while the 9070 XT
// sat empty (2026-10-11).
func TestALoadTakesAFreeGPUBeforeEvicting(t *testing.T) {
	cuda, vulkan := elevenCards()
	skipUnlessOpencotiServesBoth(t, cuda, vulkan)
	ctx, done := context.WithTimeout(t.Context(), 5*time.Second)
	defer done()
	s := InitScheduler(ctx)
	gpus := []ml.DeviceInfo{cuda, vulkan}
	loadedOn(s, "on-the-3090", time.Minute, cuda)
	req := newScenarioRequest(t, ctx, "second-model", 10, nil, nil).req

	busy := s.busyGPUs(gpus)
	got := freeGPUFirst(req, gpus, busy)
	if len(got) != 1 || got[0].PCIID != vulkan.PCIID {
		t.Fatalf("free GPU = %v, want the 9070 XT", ids(got))
	}

	// The card's other backend is the same card: a runner on the 3090's CUDA
	// side leaves its Vulkan copy no freer.
	copy3090 := cuda
	copy3090.DeviceID = ml.DeviceID{ID: "1", Library: "Vulkan"}
	copy3090.FreeMemory = 24 << 30
	if got := freeGPUFirst(req, []ml.DeviceInfo{copy3090}, busy); got != nil {
		t.Errorf("the busy card's Vulkan copy was taken as free: %v", ids(got))
	}

	// Every card holds a runner: eviction is the answer.
	loadedOn(s, "on-the-9070", time.Minute, vulkan)
	if got := freeGPUFirst(req, gpus, s.busyGPUs(gpus)); got != nil {
		t.Errorf("with a runner on every card the load took %v, want an eviction", ids(got))
	}
}

func TestAFreeGPUIsNotTakenWhenItIsTheWrongAnswer(t *testing.T) {
	cuda, vulkan := elevenCards()
	skipUnlessOpencotiServesBoth(t, cuda, vulkan)
	ctx, done := context.WithTimeout(t.Context(), 5*time.Second)
	defer done()
	s := InitScheduler(ctx)
	loadedOn(s, "on-the-3090", time.Minute, cuda)
	req := newScenarioRequest(t, ctx, "second-model", 10, nil, nil).req
	gpus := []ml.DeviceInfo{cuda, vulkan}
	busy := s.busyGPUs(gpus)

	small := vulkan
	small.FreeMemory = modelBytes(req.model)*freeGPUFloorPercent/100 - 1
	if got := freeGPUFirst(req, []ml.DeviceInfo{cuda, small}, busy); got != nil {
		t.Errorf("a card too small for most of the model was taken: %v", ids(got))
	}
	igpu := vulkan
	igpu.Integrated = true
	if got := freeGPUFirst(req, []ml.DeviceInfo{cuda, igpu}, busy); got != nil {
		t.Errorf("an integrated GPU was taken: %v", ids(got))
	}
	main := 0
	pinned := *req
	pinned.opts.MainGPU = &main
	if got := freeGPUFirst(&pinned, gpus, busy); got != nil {
		t.Errorf("a request naming its main_gpu was moved: %v", ids(got))
	}

	// Off means off: stock llama.cpp keeps upstream's eviction.
	t.Setenv("XOLLAMA_ENGINE", "llamacpp")
	if got := freeGPUFirst(req, gpus, busy); got != nil {
		t.Errorf("with XOLLAMA_ENGINE=llamacpp the load took %v, want upstream's eviction", ids(got))
	}
	if got := s.findRunnerToUnloadOn(req, []ml.DeviceInfo{vulkan}, busy); got == nil || got.modelKey != "on-the-3090" {
		t.Errorf("with XOLLAMA_ENGINE=llamacpp the victim is not upstream's: %v", got)
	}
}

// The scheduler does it: the first try asks for a full fit and is told to
// evict, the second goes to the free card with no such demand, and the model
// on the 3090 stays loaded.
func TestSchedLoadsOnTheFreeGPUAndKeepsTheLoadedModel(t *testing.T) {
	cuda, vulkan := elevenCards()
	skipUnlessOpencotiServesBoth(t, cuda, vulkan)
	ctx, done := context.WithTimeout(t.Context(), 5*time.Second)
	defer done()
	s := InitScheduler(ctx)
	s.waitForRecovery = 10 * time.Millisecond
	s.getGpuFn = func(context.Context, []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
		return []ml.DeviceInfo{cuda, vulkan}
	}
	s.getSystemInfoFn = getSystemInfoFn
	kept := loadedOn(s, "on-the-3090", time.Minute, cuda)

	type call struct {
		gpus        []ml.DeviceInfo
		requireFull bool
	}
	calls := make(chan call, 4)
	s.loadFn = func(req *LlmRequest, _ ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) bool {
		calls <- call{gpus, requireFull}
		if requireFull {
			return true // upstream's prediction: does not fit, evict
		}
		req.errCh <- context.Canceled // end the request; the calls are what is checked
		return false
	}
	a := newScenarioRequest(t, ctx, "second-model", 10, &api.Duration{Duration: 5 * time.Millisecond}, nil)
	s.pendingReqCh <- a.req
	s.Run(ctx)

	first, second := <-calls, <-calls
	if !first.requireFull || len(first.gpus) != 2 {
		t.Fatalf("first try = %v full=%v, want both cards with a full fit", ids(first.gpus), first.requireFull)
	}
	if second.requireFull || len(second.gpus) != 1 || second.gpus[0].PCIID != vulkan.PCIID {
		t.Fatalf("second try = %v full=%v, want the 9070 XT alone, no full-fit demand", ids(second.gpus), second.requireFull)
	}
	<-a.req.errCh
	s.loadedMu.Lock()
	still := s.loaded["on-the-3090"] == kept
	s.loadedMu.Unlock()
	kept.refMu.Lock()
	expiring := kept.sessionDuration == 0
	kept.refMu.Unlock()
	if !still || expiring {
		t.Fatal("the model on the 3090 was evicted although the 9070 XT was free")
	}
}

// When something has to go, it is a runner on the cards the load can use: the
// one idle longest on another card frees nothing for it.
func TestTheEvictedRunnerIsOnTheLoadsGPUs(t *testing.T) {
	cuda, vulkan := elevenCards()
	skipUnlessOpencotiServesBoth(t, cuda, vulkan)
	ctx, done := context.WithTimeout(t.Context(), 5*time.Second)
	defer done()
	s := InitScheduler(ctx)
	loadedOn(s, "a-on-the-3090", time.Second, cuda) // upstream's pick: shortest duration, first by name
	loadedOn(s, "b-on-the-9070", time.Hour, vulkan)
	req := newScenarioRequest(t, ctx, "pinned-to-the-9070", 10, nil, nil).req
	busy := s.busyGPUs([]ml.DeviceInfo{cuda, vulkan})

	if got := s.findRunnerToUnload(); got == nil || got.modelKey != "a-on-the-3090" {
		t.Fatalf("upstream's victim = %v; the test no longer tells the two apart", got)
	}
	if got := s.findRunnerToUnloadOn(req, []ml.DeviceInfo{vulkan}, busy); got == nil || got.modelKey != "b-on-the-9070" {
		t.Errorf("victim = %v, want the runner on the 9070 XT", got)
	}
	// A card with no runner: upstream decides.
	other := vulkan
	other.PCIID = "0000:09:00.0"
	if got := s.findRunnerToUnloadOn(req, []ml.DeviceInfo{other}, busy); got == nil || got.modelKey != "a-on-the-3090" {
		t.Errorf("victim for a card no runner is on = %v, want upstream's", got)
	}
}
