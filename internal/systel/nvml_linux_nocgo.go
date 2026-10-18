//go:build linux && !cgo

package systel

// Without cgo there is no dlopen, so no NVML: NVIDIA GPUs are then read from
// sysfs, for their link only.
func loadNVML() *nvmlLib { return nil }
