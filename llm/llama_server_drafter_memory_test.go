package llm

import (
	"io"
	"testing"
)

// The lines opencoti c13 r2 wrote for Gemma 4 A4B with its assistant head on
// a free 3090 (solidPC, 2026-10-15, /srv/ml/c41/run/serve.log), in order: the
// target, then the head loaded into it. The head's load_tensors and
// sched_reserve lines replaced the target's, and `ps` said 2.0 GB.
const targetThenHead = `llama_model_loader: loaded meta data with 50 key-value pairs and 658 tensors from /store/blobs/sha256-target (version GGUF V3 (latest))
load_tensors:   CPU_Mapped model buffer size =   577.50 MiB
load_tensors:        CUDA0 model buffer size = 12611.38 MiB
llama_kv_cache:      CUDA0 KV buffer size =  1000.00 MiB
sched_reserve:      CUDA0 compute buffer size =   342.06 MiB
srv    load_model: loading MTP assistant '/store/blobs/sha256-head' into target
llama_model_loader: loaded meta data with 52 key-value pairs and 49 tensors from /store/blobs/sha256-head (version GGUF V3 (latest))
load_tensors:   CPU_Mapped model buffer size =   272.00 MiB
load_tensors:        CUDA0 model buffer size =   425.34 MiB
sched_reserve:      CUDA0 compute buffer size =   211.02 MiB
`

func TestADrafterIsCountedBesideItsTargetNotInsteadOfIt(t *testing.T) {
	mib := func(f float64) uint64 { return uint64(f * 1024 * 1024) }
	wantGPU := mib(12611.38) + mib(1000) + mib(342.06) + mib(425.34) + mib(211.02)

	// Line by line and in one write: a pipe may deliver either.
	for name, writes := range map[string][]string{
		"one write": {targetThenHead},
		"per line":  splitLines(targetThenHead),
	} {
		runner := &llamaServerRunner{}
		w := &memoryParsingWriter{inner: io.Discard, runner: runner}
		for _, b := range writes {
			if _, err := w.Write([]byte(b)); err != nil {
				t.Fatal(err)
			}
		}
		if runner.memGPU != wantGPU {
			t.Fatalf("%s: GPU = %.0f MiB, want %.0f (the target's buffers and the head's)", name,
				float64(runner.memGPU)/(1<<20), float64(wantGPU)/(1<<20))
		}
	}
}

// A restatement still replaces, within one model: opencoti's second sizing
// pass of the target's KV.
func TestATargetRestatingItsKVStillReplacesIt(t *testing.T) {
	runner := &llamaServerRunner{}
	w := &memoryParsingWriter{inner: io.Discard, runner: runner}
	w.Write([]byte(targetThenHead))
	w.Write([]byte("llama_model_loader: loaded meta data with 50 key-value pairs and 658 tensors from /store/blobs/sha256-target (version GGUF V3 (latest))\n" +
		"llama_kv_cache:      CUDA0 KV buffer size =  2250.00 MiB\n"))
	mib := func(f float64) uint64 { return uint64(f * 1024 * 1024) }
	want := mib(12611.38) + mib(2250) + mib(342.06) + mib(425.34) + mib(211.02)
	if runner.memGPU != want {
		t.Fatalf("GPU = %.0f MiB, want %.0f", float64(runner.memGPU)/(1<<20), float64(want)/(1<<20))
	}
}

func splitLines(s string) []string {
	var out []string
	start := 0
	for i := range len(s) {
		if s[i] == '\n' {
			out = append(out, s[start:i+1])
			start = i + 1
		}
	}
	return out
}
