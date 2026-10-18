package systel

import (
	"bufio"
	"bytes"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// linuxPlatform reads /proc. procfs is the root to read, for tests.
type linuxPlatform struct{ procfs string }

// userHZ is the unit of /proc's processor times. The kernel exports them in
// USER_HZ, which is 100 on every architecture Linux supports today.
const userHZ = 100

func (l linuxPlatform) host() hostInfo {
	var h hostInfo
	if b, err := os.ReadFile(filepath.Join(l.procfs, "stat")); err == nil {
		line, _, _ := bytes.Cut(b, []byte("\n"))
		f := strings.Fields(string(line))
		if len(f) > 5 && f[0] == "cpu" {
			var total, idle float64
			for i, v := range f[1:] {
				n, _ := strconv.ParseFloat(v, 64)
				// guest and guest_nice are already counted in user and nice.
				if i >= 8 {
					break
				}
				total += n
				if i == 3 || i == 4 { // idle, iowait
					idle += n
				}
			}
			h.busy, h.total, h.ok = total-idle, total, true
		}
	}
	if b, err := os.ReadFile(filepath.Join(l.procfs, "uptime")); err == nil {
		if f := strings.Fields(string(b)); len(f) > 0 {
			if v, err := strconv.ParseFloat(f[0], 64); err == nil {
				h.uptime = ptr(uint64(v))
			}
		}
	}
	if b, err := os.ReadFile(filepath.Join(l.procfs, "loadavg")); err == nil {
		f := strings.Fields(string(b))
		if len(f) >= 3 {
			for _, v := range f[:3] {
				n, err := strconv.ParseFloat(v, 64)
				if err != nil {
					h.load = nil
					break
				}
				h.load = append(h.load, n)
			}
		}
	}
	if m := meminfo(filepath.Join(l.procfs, "meminfo")); m != nil {
		if t, ok := m["MemTotal"]; ok {
			h.memTotal = ptr(t)
			if a, ok := m["MemAvailable"]; ok && a <= t {
				h.memUsed = ptr(t - a)
			}
		}
		if t, ok := m["SwapTotal"]; ok {
			h.swapTotal = ptr(t)
			if f, ok := m["SwapFree"]; ok && f <= t {
				h.swapUsed = ptr(t - f)
			}
		}
	}
	return h
}

// meminfo is /proc/meminfo in bytes.
func meminfo(path string) map[string]uint64 {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	m := map[string]uint64{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			continue
		}
		fs := strings.Fields(v)
		if len(fs) == 0 {
			continue
		}
		n, err := strconv.ParseUint(fs[0], 10, 64)
		if err != nil {
			continue
		}
		if len(fs) > 1 && fs[1] == "kB" {
			n *= 1024
		}
		m[k] = n
	}
	return m
}

func (l linuxPlatform) procs() []procInfo {
	ents, err := os.ReadDir(l.procfs)
	if err != nil {
		return nil
	}
	page := uint64(os.Getpagesize())
	out := make([]procInfo, 0, len(ents))
	for _, e := range ents {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		b, err := os.ReadFile(filepath.Join(l.procfs, e.Name(), "stat"))
		if err != nil {
			continue // exited since the listing
		}
		if p, ok := parseStat(pid, b, page); ok {
			out = append(out, p)
		}
	}
	return out
}

// parseStat reads /proc/<pid>/stat. The name is in parentheses and may hold
// spaces and parentheses itself, so the fields are counted from the last ')'.
func parseStat(pid int, b []byte, page uint64) (procInfo, bool) {
	open := bytes.IndexByte(b, '(')
	end := bytes.LastIndexByte(b, ')')
	if open < 0 || end < open {
		return procInfo{}, false
	}
	f := strings.Fields(string(b[end+1:]))
	// f[0] is field 3 (state); ppid is field 4, utime 14, stime 15,
	// starttime 22, rss 24.
	if len(f) < 22 {
		return procInfo{}, false
	}
	num := func(i int) uint64 { n, _ := strconv.ParseUint(f[i-3], 10, 64); return n }
	ppid, _ := strconv.Atoi(f[1])
	return procInfo{
		pid:   pid,
		ppid:  ppid,
		name:  string(b[open+1 : end]),
		cpu:   float64(num(14)+num(15)) / userHZ,
		rss:   num(24) * page,
		start: num(22),
	}, true
}

// name names a program run through cosmopolitan's loader (".ape-1.10", as
// the opencoti engine is) after the program itself: the loader is argv[0]
// and the program argv[1].
func (l linuxPlatform) name(p procInfo) string {
	if !strings.HasPrefix(p.name, ".ape-") {
		return p.name
	}
	b, err := os.ReadFile(filepath.Join(l.procfs, strconv.Itoa(p.pid), "cmdline"))
	if err != nil {
		return p.name
	}
	args := strings.Split(string(b), "\x00")
	if len(args) < 2 || args[1] == "" {
		return p.name
	}
	return filepath.Base(args[1])
}
