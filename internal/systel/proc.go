package systel

import (
	"os"
	"runtime"
	"slices"
	"time"

	"github.com/ollama/ollama/api"
)

// procInfo is one process as the platform lists it.
type procInfo struct {
	pid, ppid int
	name      string
	// cpu is the processor time it has used, user and system, in seconds.
	cpu float64
	rss uint64
	// start orders a parent before its children, in any unit the platform
	// keeps: a process id is reused, and a parent started after the process
	// that names it is not its parent.
	start uint64
}

// hostInfo is the machine-wide part, as the platform reads it.
type hostInfo struct {
	busy, total float64 // processor time, in any unit, summed over every core
	uptime      *uint64
	load        []float64
	memTotal    *uint64
	memUsed     *uint64
	swapTotal   *uint64
	swapUsed    *uint64
	ok          bool // busy and total were read
}

// platform is what each operating system supplies.
type platform interface {
	host() hostInfo
	procs() []procInfo
}

// namer is a platform that can name a process better than its listing does,
// asked only for the processes in the server's tree.
type namer interface {
	name(p procInfo) string
}

// procReader fills Host and Server.
type procReader struct {
	root int
	p    platform

	prevBusy, prevTotal float64
	prevHost            bool
	prevCPU             map[int]procSample
}

type procSample struct {
	cpu   float64
	start uint64
	at    time.Time
}

func (r *procReader) reset() {
	r.prevHost = false
	r.prevCPU = nil
}

func (r *procReader) read(s *Snapshot) {
	h := r.p.host()
	s.Host = api.SystemHost{
		OS:        runtime.GOOS,
		Arch:      runtime.GOARCH,
		Hostname:  hostname(),
		UptimeS:   h.uptime,
		CPUCount:  runtime.NumCPU(),
		Load:      h.load,
		MemTotal:  h.memTotal,
		MemUsed:   h.memUsed,
		SwapTotal: h.swapTotal,
		SwapUsed:  h.swapUsed,
	}
	if h.ok {
		s.Host.CPUPct = pct(r.prevBusy, h.busy, r.prevTotal, h.total, r.prevHost)
		r.prevBusy, r.prevTotal, r.prevHost = h.busy, h.total, true
	}

	tree := subtree(r.p.procs(), r.root)
	if n, ok := r.p.(namer); ok {
		for i := range tree {
			tree[i].name = n.name(tree[i])
		}
	}
	s.Server = api.SystemServer{PID: r.root, Procs: len(tree), Children: []api.SystemProcess{}}
	next := make(map[int]procSample, len(tree))
	// The sums are reported only when every process in the tree has its
	// value: one that started since the last reading has no rate yet, and a
	// sum without it would be an undercount.
	var cpuSum, rssSum float64
	cpuKnown, rssKnown := len(tree) > 0, len(tree) > 0
	for _, p := range tree {
		next[p.pid] = procSample{cpu: p.cpu, start: p.start, at: s.At}
		var cpu *float64
		if prev, ok := r.prevCPU[p.pid]; ok && prev.start == p.start {
			if dt := s.At.Sub(prev.at).Seconds(); dt > 0 && p.cpu >= prev.cpu {
				cpu = ptr(100 * (p.cpu - prev.cpu) / dt)
			}
		}
		if cpu != nil {
			cpuSum += *cpu
		} else {
			cpuKnown = false
		}
		var rss *uint64
		if p.rss > 0 {
			rss = ptr(p.rss)
			rssSum += float64(p.rss)
		} else {
			rssKnown = false
		}
		if p.pid != r.root {
			s.Server.Children = append(s.Server.Children, api.SystemProcess{PID: p.pid, Name: p.name, CPUPct: cpu, RSS: rss})
		}
	}
	if cpuKnown {
		s.Server.CPUPct = ptr(cpuSum)
	}
	if rssKnown {
		s.Server.RSS = ptr(uint64(rssSum))
	}
	r.prevCPU = next
}

// subtree is root and every process under it, root first.
func subtree(all []procInfo, root int) []procInfo {
	byPID := make(map[int]procInfo, len(all))
	children := make(map[int][]int)
	for _, p := range all {
		byPID[p.pid] = p
		children[p.ppid] = append(children[p.ppid], p.pid)
	}
	r, ok := byPID[root]
	if !ok {
		return nil
	}
	out := []procInfo{r}
	for i := 0; i < len(out); i++ {
		parent := out[i]
		kids := children[parent.pid]
		slices.Sort(kids)
		for _, pid := range kids {
			c := byPID[pid]
			if pid == parent.pid || c.start < parent.start {
				continue
			}
			out = append(out, c)
		}
	}
	return out
}

func hostname() string {
	h, _ := os.Hostname()
	return h
}
