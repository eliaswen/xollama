//go:build linux || windows

package systel

import (
	"bytes"
	"encoding/binary"
	"log/slog"
	"strings"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/api"
)

// NVML, loaded at run time from the driver. The library is opened once and
// kept; each reading is a handful of calls of microseconds each.
//
// nvmlLib is the per-OS part: the symbol table and a call with up to three
// word-sized arguments. Every pointer argument is into scratch, memory outside
// Go's heap, so nothing Go owns is handed to C.
type nvmlLib struct {
	sym     map[string]uintptr
	call    func(fn uintptr, a ...uintptr) int
	scratch unsafe.Pointer // scratchSize bytes
}

const scratchSize = 256

var (
	nvmlOnce sync.Once
	nvml     *nvmlLib
)

// nvmlOpen loads NVML and initialises it, once. Without a driver it is nil.
func nvmlOpen() *nvmlLib {
	nvmlOnce.Do(func() {
		l := loadNVML()
		if l == nil {
			return
		}
		if rc := l.fn("nvmlInit_v2"); rc != 0 {
			slog.Debug("system telemetry: nvmlInit_v2 failed", "rc", rc)
			return
		}
		nvml = l
	})
	return nvml
}

// fn calls a symbol, or answers not-supported (3) when the driver lacks it.
func (l *nvmlLib) fn(name string, a ...uintptr) int {
	f := l.sym[name]
	if f == 0 {
		return 3
	}
	return l.call(f, a...)
}

func (l *nvmlLib) buf() []byte { return unsafe.Slice((*byte)(l.scratch), scratchSize) }

func (l *nvmlLib) at() uintptr { return uintptr(l.scratch) }

// u32 calls fn(device, &out) and reads a uint32.
func (l *nvmlLib) u32(name string, args ...uintptr) *uint32 {
	clear(l.buf())
	if l.fn(name, append(args, l.at())...) != 0 {
		return nil
	}
	return ptr(binary.LittleEndian.Uint32(l.buf()))
}

type nvmlReader struct{}

func (nvmlReader) reset() {}

func (nvmlReader) read(s *Snapshot) {
	l := nvmlOpen()
	if l == nil {
		return
	}
	n := l.u32("nvmlDeviceGetCount_v2")
	if n == nil {
		return
	}
	for i := range *n {
		clear(l.buf())
		if l.fn("nvmlDeviceGetHandleByIndex_v2", uintptr(i), l.at()) != 0 {
			continue
		}
		dev := uintptr(binary.LittleEndian.Uint64(l.buf()))
		s.Devices = append(s.Devices, l.device(dev))
	}
}

func (l *nvmlLib) device(dev uintptr) api.SystemDevice {
	d := api.SystemDevice{Vendor: "nvidia", Source: "nvml", Link: &api.SystemLink{}}

	clear(l.buf())
	if l.fn("nvmlDeviceGetName", dev, l.at(), 96) == 0 {
		d.Name = cstring(l.buf()[:96])
	}
	// nvmlPciInfo_t: busIdLegacy[16], domain, bus, device, pciDeviceId,
	// pciSubSystemId (uint32 each), busId[32].
	clear(l.buf())
	if l.fn("nvmlDeviceGetPciInfo_v3", dev, l.at()) == 0 {
		d.PCIID = normalizePCI(cstring(l.buf()[36:68]))
	}
	// nvmlMemory_t: total, free, used (uint64 each).
	clear(l.buf())
	if l.fn("nvmlDeviceGetMemoryInfo", dev, l.at()) == 0 {
		b := l.buf()
		d.TotalMemory = ptr(binary.LittleEndian.Uint64(b[0:]))
		d.FreeMemory = ptr(binary.LittleEndian.Uint64(b[8:]))
		d.UsedMemory = ptr(binary.LittleEndian.Uint64(b[16:]))
	}
	// nvmlUtilization_t: gpu, memory (uint32 each, percent).
	clear(l.buf())
	if l.fn("nvmlDeviceGetUtilizationRates", dev, l.at()) == 0 {
		b := l.buf()
		d.UtilPct = ptr(float64(binary.LittleEndian.Uint32(b[0:])))
		d.MemUtilPct = ptr(float64(binary.LittleEndian.Uint32(b[4:])))
	}
	if v := l.u32("nvmlDeviceGetTemperature", dev, 0); v != nil { // NVML_TEMPERATURE_GPU
		d.TempC = ptr(float64(*v))
	}
	if v := l.u32("nvmlDeviceGetPowerUsage", dev); v != nil {
		d.PowerW = ptr(float64(*v) / 1000)
	}
	if v := l.u32("nvmlDeviceGetEnforcedPowerLimit", dev); v != nil {
		d.PowerLimitW = ptr(float64(*v) / 1000)
	}
	if v := l.u32("nvmlDeviceGetClockInfo", dev, 0); v != nil { // NVML_CLOCK_GRAPHICS
		d.ClockMHz = ptr(uint64(*v))
	}
	if v := l.u32("nvmlDeviceGetMaxClockInfo", dev, 0); v != nil {
		d.ClockMaxMHz = ptr(uint64(*v))
	}
	if v := l.u32("nvmlDeviceGetFanSpeed", dev); v != nil {
		d.FanPct = ptr(float64(*v))
	}
	link := func(name string) *int {
		if v := l.u32(name, dev); v != nil {
			return ptr(int(*v))
		}
		return nil
	}
	d.Link.Gen = link("nvmlDeviceGetCurrPcieLinkGeneration")
	d.Link.Width = link("nvmlDeviceGetCurrPcieLinkWidth")
	d.Link.MaxGen = link("nvmlDeviceGetMaxPcieLinkGeneration")
	d.Link.MaxWidth = link("nvmlDeviceGetMaxPcieLinkWidth")
	return d
}

var nvmlSymbols = []string{
	"nvmlInit_v2", "nvmlDeviceGetCount_v2", "nvmlDeviceGetHandleByIndex_v2",
	"nvmlDeviceGetName", "nvmlDeviceGetPciInfo_v3", "nvmlDeviceGetMemoryInfo",
	"nvmlDeviceGetUtilizationRates", "nvmlDeviceGetTemperature",
	"nvmlDeviceGetPowerUsage", "nvmlDeviceGetEnforcedPowerLimit",
	"nvmlDeviceGetClockInfo", "nvmlDeviceGetMaxClockInfo", "nvmlDeviceGetFanSpeed",
	"nvmlDeviceGetCurrPcieLinkGeneration", "nvmlDeviceGetCurrPcieLinkWidth",
	"nvmlDeviceGetMaxPcieLinkGeneration", "nvmlDeviceGetMaxPcieLinkWidth",
}

func cstring(b []byte) string {
	if i := bytes.IndexByte(b, 0); i >= 0 {
		b = b[:i]
	}
	return strings.TrimSpace(string(b))
}
