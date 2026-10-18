package systel

import (
	"strings"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	kernel32                 = windows.NewLazySystemDLL("kernel32.dll")
	ntdll                    = windows.NewLazySystemDLL("ntdll.dll")
	procGetSystemTimes       = kernel32.NewProc("GetSystemTimes")
	procGlobalMemoryStatusEx = kernel32.NewProc("GlobalMemoryStatusEx")
	procGetTickCount64       = kernel32.NewProc("GetTickCount64")
	procK32GetProcessMemInfo = kernel32.NewProc("K32GetProcessMemoryInfo")
	procNtQuerySystemInfo    = ntdll.NewProc("NtQuerySystemInformation")
)

type windowsPlatform struct{}

func filetime(f windows.Filetime) float64 {
	return float64(uint64(f.HighDateTime)<<32 | uint64(f.LowDateTime))
}

func (windowsPlatform) host() hostInfo {
	var h hostInfo
	var idle, kern, user windows.Filetime
	if r, _, _ := procGetSystemTimes.Call(uintptr(unsafe.Pointer(&idle)), uintptr(unsafe.Pointer(&kern)), uintptr(unsafe.Pointer(&user))); r != 0 {
		// Kernel time includes the idle time.
		total := filetime(kern) + filetime(user)
		h.busy, h.total, h.ok = total-filetime(idle), total, true
	}
	if r, _, _ := procGetTickCount64.Call(); r != 0 {
		h.uptime = ptr(uint64(r) / 1000)
	}
	// MEMORYSTATUSEX: dwLength, dwMemoryLoad, then ullTotalPhys, ullAvailPhys…
	var ms [64]byte
	*(*uint32)(unsafe.Pointer(&ms[0])) = uint32(len(ms))
	if r, _, _ := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&ms[0]))); r != 0 {
		total := *(*uint64)(unsafe.Pointer(&ms[8]))
		avail := *(*uint64)(unsafe.Pointer(&ms[16]))
		h.memTotal = ptr(total)
		if avail <= total {
			h.memUsed = ptr(total - avail)
		}
	}
	h.swapTotal, h.swapUsed = pagefiles()
	return h
}

// pagefiles adds up the paging files, the Windows counterpart of swap.
func pagefiles() (total, used *uint64) {
	const systemPageFileInformation = 18
	buf := make([]byte, 4096)
	var n uint32
	if r, _, _ := procNtQuerySystemInfo.Call(systemPageFileInformation, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)), uintptr(unsafe.Pointer(&n))); r != 0 {
		return nil, nil
	}
	page := uint64(windows.Getpagesize())
	var t, u uint64
	// SYSTEM_PAGEFILE_INFORMATION: NextEntryOffset, TotalSize, TotalInUse,
	// PeakUsage (pages, uint32 each), then the file's name.
	for off := uint32(0); n >= 16 && off+16 <= n; {
		e := buf[off:]
		t += uint64(*(*uint32)(unsafe.Pointer(&e[4]))) * page
		u += uint64(*(*uint32)(unsafe.Pointer(&e[8]))) * page
		next := *(*uint32)(unsafe.Pointer(&e[0]))
		if next == 0 {
			break
		}
		off += next
	}
	return &t, &u
}

func (windowsPlatform) procs() []procInfo {
	snap, err := windows.CreateToolhelp32Snapshot(windows.TH32CS_SNAPPROCESS, 0)
	if err != nil {
		return nil
	}
	defer windows.CloseHandle(snap)
	var all []procInfo
	var e windows.ProcessEntry32
	e.Size = uint32(unsafe.Sizeof(e))
	for err = windows.Process32First(snap, &e); err == nil; err = windows.Process32Next(snap, &e) {
		all = append(all, procInfo{pid: int(e.ProcessID), ppid: int(e.ParentProcessID), name: windows.UTF16ToString(e.ExeFile[:])})
	}
	// Open only what may be in the tree; the start times then decide.
	root := windows.GetCurrentProcessId()
	cand := subtree(all, int(root))
	out := make([]procInfo, 0, len(cand))
	for _, p := range cand {
		h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(p.pid))
		if err != nil {
			continue
		}
		var created, exited, kern, user windows.Filetime
		if windows.GetProcessTimes(h, &created, &exited, &kern, &user) == nil {
			p.start = uint64(filetime(created))
			p.cpu = (filetime(kern) + filetime(user)) / 1e7 // 100 ns units
		}
		// PROCESS_MEMORY_COUNTERS: cb, PageFaultCount, PeakWorkingSetSize,
		// WorkingSetSize…
		var pmc [72]byte
		*(*uint32)(unsafe.Pointer(&pmc[0])) = uint32(len(pmc))
		if r, _, _ := procK32GetProcessMemInfo.Call(uintptr(h), uintptr(unsafe.Pointer(&pmc[0])), uintptr(len(pmc))); r != 0 {
			p.rss = *(*uint64)(unsafe.Pointer(&pmc[16]))
		}
		windows.CloseHandle(h)
		p.name = strings.TrimSuffix(p.name, ".exe")
		out = append(out, p)
	}
	return out
}
