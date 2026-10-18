package systel

/*
#include <libproc.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <sys/sysctl.h>

static int systel_cpu(unsigned long long *busy, unsigned long long *total) {
	host_cpu_load_info_data_t info;
	mach_msg_type_number_t n = HOST_CPU_LOAD_INFO_COUNT;
	if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, (host_info_t)&info, &n) != KERN_SUCCESS) return -1;
	unsigned long long u = info.cpu_ticks[CPU_STATE_USER], s = info.cpu_ticks[CPU_STATE_SYSTEM],
		i = info.cpu_ticks[CPU_STATE_IDLE], ni = info.cpu_ticks[CPU_STATE_NICE];
	*busy = u + s + ni;
	*total = u + s + ni + i;
	return 0;
}

// Memory used as Activity Monitor counts it: app memory (internal less
// purgeable), wired and compressed.
static int systel_mem_used(unsigned long long *used) {
	vm_statistics64_data_t vm;
	mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
	if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &n) != KERN_SUCCESS) return -1;
	vm_size_t page;
	if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS) return -1;
	unsigned long long app = vm.internal_page_count > vm.purgeable_count ? vm.internal_page_count - vm.purgeable_count : 0;
	*used = (app + vm.wire_count + vm.compressor_page_count) * (unsigned long long)page;
	return 0;
}

static int systel_swap(unsigned long long *total, unsigned long long *used) {
	struct xsw_usage sw;
	size_t len = sizeof(sw);
	if (sysctlbyname("vm.swapusage", &sw, &len, NULL, 0) != 0) return -1;
	*total = sw.xsu_total;
	*used = sw.xsu_used;
	return 0;
}

static int systel_pids(int *buf, int n) { return proc_listallpids(buf, n * (int)sizeof(int)); }

static int systel_bsd(int pid, int *ppid, char *name, unsigned long long *start) {
	struct proc_bsdinfo b;
	if (proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &b, sizeof(b)) != sizeof(b)) return -1;
	*ppid = (int)b.pbi_ppid;
	strlcpy(name, b.pbi_comm, MAXCOMLEN + 1);
	*start = (unsigned long long)b.pbi_start_tvsec * 1000000ULL + b.pbi_start_tvusec;
	return 0;
}

// Processor time in nanoseconds: the task counters are in mach time units,
// which are nanoseconds on Intel and not on Apple silicon.
static int systel_task(int pid, unsigned long long *ns, unsigned long long *rss) {
	struct proc_taskinfo t;
	if (proc_pidinfo(pid, PROC_PIDTASKINFO, 0, &t, sizeof(t)) != sizeof(t)) return -1;
	static mach_timebase_info_data_t tb;
	if (tb.denom == 0) mach_timebase_info(&tb);
	*ns = (t.pti_total_user + t.pti_total_system) * tb.numer / tb.denom;
	*rss = t.pti_resident_size;
	return 0;
}
*/
import "C"

import (
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

type darwinPlatform struct{}

func (darwinPlatform) host() hostInfo {
	var h hostInfo
	var busy, total C.ulonglong
	if C.systel_cpu(&busy, &total) == 0 {
		h.busy, h.total, h.ok = float64(busy), float64(total), true
	}
	if tv, err := unix.SysctlTimeval("kern.boottime"); err == nil {
		h.uptime = ptr(uint64(time.Since(time.Unix(tv.Unix())).Seconds()))
	}
	var load [3]C.double
	if C.getloadavg(&load[0], 3) == 3 {
		h.load = []float64{float64(load[0]), float64(load[1]), float64(load[2])}
	}
	if m, err := unix.SysctlUint64("hw.memsize"); err == nil {
		h.memTotal = ptr(m)
	}
	var used C.ulonglong
	if C.systel_mem_used(&used) == 0 {
		h.memUsed = ptr(uint64(used))
	}
	var st, su C.ulonglong
	if C.systel_swap(&st, &su) == 0 {
		h.swapTotal, h.swapUsed = ptr(uint64(st)), ptr(uint64(su))
	}
	return h
}

func (darwinPlatform) procs() []procInfo {
	n := C.systel_pids(nil, 0)
	if n <= 0 {
		return nil
	}
	pids := make([]C.int, n+64)
	n = C.systel_pids(&pids[0], C.int(len(pids)))
	if n <= 0 {
		return nil
	}
	var all []procInfo
	name := (*C.char)(C.malloc(C.MAXCOMLEN + 1))
	defer C.free(unsafe.Pointer(name))
	for _, pid := range pids[:min(int(n), len(pids))] {
		var ppid C.int
		var start C.ulonglong
		if pid == 0 || C.systel_bsd(pid, &ppid, name, &start) != 0 {
			continue
		}
		all = append(all, procInfo{pid: int(pid), ppid: int(ppid), name: C.GoString(name), start: uint64(start)})
	}
	// The counters only for what may be in the tree.
	cand := subtree(all, unix.Getpid())
	for i := range cand {
		var ns, rss C.ulonglong
		if C.systel_task(C.int(cand[i].pid), &ns, &rss) == 0 {
			cand[i].cpu = float64(ns) / 1e9
			cand[i].rss = uint64(rss)
		}
	}
	return cand
}
