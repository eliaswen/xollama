package systel

import (
	"os"
	"path/filepath"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

func loadNVML() *nvmlLib {
	h, err := windows.LoadLibraryEx("nvml.dll", 0, windows.LOAD_LIBRARY_SEARCH_SYSTEM32)
	if err != nil {
		// Older drivers install it beside nvidia-smi only.
		p := filepath.Join(os.Getenv("ProgramFiles"), "NVIDIA Corporation", "NVSMI", "nvml.dll")
		if h, err = windows.LoadLibraryEx(p, 0, windows.LOAD_WITH_ALTERED_SEARCH_PATH); err != nil {
			return nil
		}
	}
	mem, err := windows.LocalAlloc(windows.LPTR, scratchSize)
	if err != nil {
		return nil
	}
	l := &nvmlLib{sym: map[string]uintptr{}, scratch: *(*unsafe.Pointer)(unsafe.Pointer(&mem))}
	for _, name := range nvmlSymbols {
		if p, err := windows.GetProcAddress(h, name); err == nil {
			l.sym[name] = p
		}
	}
	l.call = func(fn uintptr, a ...uintptr) int {
		r, _, _ := syscall.SyscallN(fn, a...)
		return int(int32(r))
	}
	return l
}
