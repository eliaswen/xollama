package systel

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>

static void *systel_dlopen(const char *path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
static void *systel_dlsym(void *h, const char *name) { return dlsym(h, name); }

// Every NVML entry used here returns int and takes at most three word-sized
// arguments; the SysV calling conventions pass them in registers, so a
// callee with fewer parameters ignores the rest.
typedef int (*systel_fn3)(uintptr_t, uintptr_t, uintptr_t);
static int systel_call3(void *f, uintptr_t a, uintptr_t b, uintptr_t c) { return ((systel_fn3)f)(a, b, c); }
*/
import "C"

import (
	"unsafe"
)

func loadNVML() *nvmlLib {
	var h unsafe.Pointer
	for _, name := range []string{"libnvidia-ml.so.1", "libnvidia-ml.so"} {
		cs := C.CString(name)
		h = C.systel_dlopen(cs)
		C.free(unsafe.Pointer(cs))
		if h != nil {
			break
		}
	}
	if h == nil {
		return nil
	}
	l := &nvmlLib{sym: map[string]uintptr{}, scratch: C.calloc(1, scratchSize)}
	for _, name := range nvmlSymbols {
		cs := C.CString(name)
		if p := C.systel_dlsym(h, cs); p != nil {
			l.sym[name] = uintptr(p)
		}
		C.free(unsafe.Pointer(cs))
	}
	l.call = func(fn uintptr, a ...uintptr) int {
		var args [3]uintptr
		copy(args[:], a)
		return int(C.systel_call3(*(*unsafe.Pointer)(unsafe.Pointer(&fn)), C.uintptr_t(args[0]), C.uintptr_t(args[1]), C.uintptr_t(args[2])))
	}
	return l
}
