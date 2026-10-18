//go:build cgo

package systel

/*
#cgo LDFLAGS: -framework IOKit -framework CoreFoundation
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

static int systel_num(CFDictionaryRef d, const char *key, long long *out) {
	CFStringRef k = CFStringCreateWithCString(NULL, key, kCFStringEncodingUTF8);
	CFNumberRef v = (CFNumberRef)CFDictionaryGetValue(d, k);
	CFRelease(k);
	if (v == NULL || CFGetTypeID(v) != CFNumberGetTypeID()) return -1;
	return CFNumberGetValue(v, kCFNumberLongLongType, out) ? 0 : -1;
}

// The GPU's statistics, as its driver publishes them in the registry: the
// same figures Activity Monitor's GPU history reads. Needs no privilege.
static int systel_gpu(int index, long long *util, long long *inuse) {
	io_iterator_t it;
	if (IOServiceGetMatchingServices(0, IOServiceMatching("IOAccelerator"), &it) != KERN_SUCCESS) return -1;
	io_object_t s;
	int i = 0, rc = -1;
	while ((s = IOIteratorNext(it))) {
		if (i++ == index) {
			CFTypeRef p = IORegistryEntryCreateCFProperty(s, CFSTR("PerformanceStatistics"), kCFAllocatorDefault, 0);
			if (p != NULL) {
				if (CFGetTypeID(p) == CFDictionaryGetTypeID()) {
					*util = -1;
					*inuse = -1;
					systel_num((CFDictionaryRef)p, "Device Utilization %", util);
					systel_num((CFDictionaryRef)p, "In use system memory", inuse);
					rc = 0;
				}
				CFRelease(p);
			}
		}
		IOObjectRelease(s);
	}
	IOObjectRelease(it);
	return rc;
}
*/
import "C"

import (
	"runtime"

	"github.com/ollama/ollama/api"
)

type iokitReader struct{}

func (iokitReader) reset() {}

func (iokitReader) read(s *Snapshot) {
	if runtime.GOARCH != "arm64" {
		return // an Intel Mac's GPUs are not what the server runs on
	}
	var util, inuse C.longlong
	if C.systel_gpu(0, &util, &inuse) != 0 {
		return
	}
	d := api.SystemDevice{Vendor: "apple", Source: "iokit", Integrated: true}
	if util >= 0 {
		d.UtilPct = ptr(float64(util))
	}
	if inuse >= 0 {
		d.UsedMemory = ptr(uint64(inuse))
	}
	s.Devices = append(s.Devices, d)
}
