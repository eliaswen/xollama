package engine

import "fmt"

// Engine capabilities that the pinned artifact may or may not have.
//
// The pin file names the exact bytes this build ships, and those bytes are the
// only authority on what argv they accept. A flag that exists in opencoti's
// development tree is not a flag this build can send: the artifact rejects the
// whole command line rather than ignoring what it does not know.
//
// So a feature whose flag the shipped artifact lacks is refused where the
// operator set it, naming the build, rather than being sent and turned into
// "error: invalid argument" from a subprocess -- which names nothing the
// operator can act on.
//
// Capabilities are DECLARED by the pin, not inferred from its tag. They used to
// be worked out from the cut number ("c8 or newer has the SWA cache types"),
// which is exactly wrong for the thing this fork now tracks: a dev artifact
// carries part of the next cut while still being tagged as the current one, so
// the flags it was pinned to test would have been refused. It is also a moving
// target, carrying some of a cut and not the rest -- a single number cannot say
// that and a list can. The pin states what the bytes do, we test the claim, and
// a claim that turns out to be wrong is a bug report rather than a silent
// mismatch.

// featureSWACacheTypes is --cache-type-k-swa / --cache-type-v-swa, which let
// the sliding-window half of the cache take a different type from the global
// half. Added by opencoti patch 0288; absent from every cut through c7.
const featureSWACacheTypes = "cache_type_swa_v1"

// HasSlidingWindowRing reports whether the engine this build ships can be told
// to quantise the sliding-window half of the cache separately.
func HasSlidingWindowRing() bool {
	pin, err := DefaultPin()
	if err != nil {
		return false
	}
	return pin.HasFeature(featureSWACacheTypes)
}

// SlidingWindowRingUnavailable explains, in the operator's terms, why the ring
// cannot be set on this build. Empty when it can.
func SlidingWindowRingUnavailable() string {
	if HasSlidingWindowRing() {
		return ""
	}
	build := "the engine this build ships"
	if pin, err := DefaultPin(); err == nil && pin.Tag != "" {
		build = fmt.Sprintf("the engine this build ships (%s, channel %s)", pin.Tag, pin.Channel)
	}
	return fmt.Sprintf("%s has no --cache-type-k-swa/--cache-type-v-swa. "+
		"Set only kv.k and kv.v, which quantise both halves together on this engine",
		build)
}

// featureSystemOne is llama.cpp's native decision scoring (b11521): one
// POST /v1/systemone per request, for the `clef` architecture and every other
// decision type. Since v0.40.4 upstream scores every GGUF decision model that
// way; the engine announces it once it serves the route (opencoti op-139).
const featureSystemOne = "systemone_v1"

// HasSystemOne reports whether the engine this build ships serves
// /v1/systemone. While it does not, decision models are served by stock
// llama-server (server/decision_engine.go).
func HasSystemOne() bool {
	pin, err := DefaultPin()
	if err != nil {
		return false
	}
	return pin.HasFeature(featureSystemOne)
}

// FeatureGenerationNotice is the engine's generation notices, contract v2
// (opencoti c13 r3, mails #1487-#1495): a request's top-level `notices` with
// per-step templates, --notice-templates FILE, --print-notice-config, and
// POST /v1/notice. Contract v1 was never pinned and is superseded.
const FeatureGenerationNotice = "generation_notice_v2"

// FeatureWindowUnheld is the engine's num_ctx_hold (opencoti c13 r3, patch
// 0683): false makes a request's num_ctx/num_ctx_min its own window, not held
// for its session afterwards.
const FeatureWindowUnheld = "elastic_window_unheld_v1"

// HasWindowUnheld reports whether the engine this build ships takes
// num_ctx_hold.
func HasWindowUnheld() bool {
	pin, err := DefaultPin()
	if err != nil {
		return false
	}
	return pin.HasFeature(FeatureWindowUnheld)
}

// HasGenerationNotice reports whether the engine this build ships plants
// generation notices. An operator's XOLLAMA_ENGINE_PATH is asked instead
// (NoticesFor), since the pin does not describe it.
func HasGenerationNotice() bool {
	pin, err := DefaultPin()
	if err != nil {
		return false
	}
	return pin.HasFeature(FeatureGenerationNotice)
}
