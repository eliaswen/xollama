package systel

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func write(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestProcfsHost(t *testing.T) {
	d := t.TempDir()
	write(t, filepath.Join(d, "stat"), "cpu  100 0 50 800 50 0 0 0 7 0\ncpu0 1 2 3 4\n")
	write(t, filepath.Join(d, "uptime"), "1234.56 999.0\n")
	write(t, filepath.Join(d, "loadavg"), "0.50 1.00 2.00 1/100 42\n")
	write(t, filepath.Join(d, "meminfo"), "MemTotal: 1000 kB\nMemFree: 100 kB\nMemAvailable: 400 kB\nSwapTotal: 200 kB\nSwapFree: 50 kB\n")
	h := linuxPlatform{procfs: d}.host()
	if !h.ok || h.busy != 150 || h.total != 1000 {
		t.Errorf("cpu busy %v total %v (guest time is already in user)", h.busy, h.total)
	}
	if *h.uptime != 1234 || len(h.load) != 3 || h.load[2] != 2 {
		t.Errorf("uptime %v load %v", *h.uptime, h.load)
	}
	if *h.memTotal != 1000<<10 || *h.memUsed != 600<<10 || *h.swapTotal != 200<<10 || *h.swapUsed != 150<<10 {
		t.Errorf("memory %v %v swap %v %v", *h.memTotal, *h.memUsed, *h.swapTotal, *h.swapUsed)
	}
}

// A process name may hold spaces and parentheses.
func TestParseStatCountsFromTheLastParenthesis(t *testing.T) {
	line := "42 (llama (server) x) S 7 42 42 0 -1 4194560 1 0 0 0 250 50 0 0 20 0 3 0 9000 1000000 300 18446744073709551615"
	p, ok := parseStat(42, []byte(line), 4096)
	if !ok || p.name != "llama (server) x" || p.ppid != 7 || p.cpu != 3 || p.start != 9000 || p.rss != 300*4096 {
		t.Fatalf("%+v %v", p, ok)
	}
}

func TestProcfsProcessRates(t *testing.T) {
	d := t.TempDir()
	stat := func(pid, ppid, ticks, start string) string {
		return pid + " (x) S " + ppid + " 0 0 0 -1 0 0 0 0 0 " + ticks + " 0 0 0 20 0 1 0 " + start + " 0 10"
	}
	write(t, filepath.Join(d, "stat"), "cpu 1 0 0 1 0 0 0 0\n")
	write(t, filepath.Join(d, "100", "stat"), stat("100", "1", "100", "10"))
	write(t, filepath.Join(d, "101", "stat"), stat("101", "100", "0", "20"))
	r := &procReader{root: 100, p: linuxPlatform{procfs: d}}
	at := time.Unix(0, 0)
	s := Snapshot{At: at}
	r.read(&s)
	if s.Server.CPUPct != nil || s.Server.Procs != 2 || len(s.Server.Children) != 1 {
		t.Fatalf("first reading: %+v", s.Server)
	}
	write(t, filepath.Join(d, "100", "stat"), stat("100", "1", "150", "10"))   // +0.5 s
	write(t, filepath.Join(d, "101", "stat"), stat("101", "100", "200", "20")) // +2 s
	s = Snapshot{At: at.Add(2 * time.Second)}
	r.read(&s)
	if s.Server.CPUPct == nil || *s.Server.CPUPct != 125 {
		t.Fatalf("tree cpu %v, want 125 (25 + 100)", s.Server.CPUPct)
	}
	if c := s.Server.Children[0]; c.CPUPct == nil || *c.CPUPct != 100 || *c.RSS != 10*uint64(os.Getpagesize()) {
		t.Errorf("child %+v", c)
	}
	// A child that started since has no rate yet, so the tree has no sum.
	write(t, filepath.Join(d, "102", "stat"), stat("102", "100", "5", "30"))
	s = Snapshot{At: at.Add(4 * time.Second)}
	r.read(&s)
	if s.Server.CPUPct != nil {
		t.Errorf("a sum missing a process is an undercount: %v", *s.Server.CPUPct)
	}
}

func amdCard(t *testing.T, sys, card, pci string, files map[string]string) {
	t.Helper()
	dev := filepath.Join(sys, "devices", pci)
	for name, body := range files {
		write(t, filepath.Join(dev, name), body)
	}
	if err := os.MkdirAll(filepath.Join(sys, "class", "drm"), 0o755); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(sys, "class", "drm", card, "dev"), "226:0")
	if err := os.Symlink(dev, filepath.Join(sys, "class", "drm", card, "device")); err != nil {
		t.Fatal(err)
	}
}

func TestSysfsAMD(t *testing.T) {
	sys := t.TempDir()
	amdCard(t, sys, "card1", "0000:03:00.0", map[string]string{
		"vendor":                      "0x1002",
		"product_name":                "Radeon RX 9070 XT",
		"gpu_busy_percent":            "37",
		"mem_info_vram_total":         "17163091968",
		"mem_info_vram_used":          "1073741824",
		"pp_dpm_sclk":                 "0: 500Mhz\n1: 2400Mhz *\n2: 2970Mhz\n",
		"current_link_speed":          "16.0 GT/s PCIe",
		"current_link_width":          "16",
		"max_link_speed":              "32.0 GT/s PCIe",
		"max_link_width":              "16",
		"hwmon/hwmon3/temp1_input":    "52000",
		"hwmon/hwmon3/power1_average": "88000000",
		"hwmon/hwmon3/power1_cap":     "304000000",
		"hwmon/hwmon3/pwm1":           "51",
	})
	// a connector is not a GPU
	write(t, filepath.Join(sys, "class", "drm", "card1-DP-1", "status"), "connected")
	amdCard(t, sys, "card2", "0000:04:00.0", map[string]string{
		"vendor":               "0x1002",
		"gpu_busy_percent":     "0",
		"power/runtime_status": "suspended",
		"current_link_speed":   "2.5 GT/s PCIe",
	})
	s := Snapshot{At: time.Unix(0, 0)}
	(&drmReader{sysfs: sys}).read(&s)
	if len(s.Devices) != 2 {
		t.Fatalf("devices %+v", s.Devices)
	}
	d := s.Devices[0]
	if d.PCIID != "0000:03:00.0" || d.Vendor != "amd" || d.Name != "Radeon RX 9070 XT" || *d.UtilPct != 37 ||
		*d.UsedMemory != 1<<30 || *d.FreeMemory != 17163091968-(1<<30) || *d.TempC != 52 || *d.PowerW != 88 ||
		*d.PowerLimitW != 304 || *d.FanPct != 20 || *d.ClockMHz != 2400 || *d.ClockMaxMHz != 2970 {
		t.Errorf("9070 XT %+v", d)
	}
	if l := d.Link; *l.Gen != 4 || *l.Width != 16 || *l.MaxGen != 5 || *l.MaxWidth != 16 {
		t.Errorf("link %+v", l)
	}
	if d := s.Devices[1]; d.UtilPct != nil || d.Link == nil || *d.Link.Gen != 1 {
		t.Errorf("a suspended GPU is not woken, only its link read: %+v", d)
	}
}

func TestTheEngineIsNamedAfterItsLlamafileNotTheLoader(t *testing.T) {
	d := t.TempDir()
	write(t, filepath.Join(d, "7", "cmdline"), "/home/x/.ape-1.10\x00/usr/lib/ollama/opencoti-llamafile-0.10.5-c13-bare.llamafile\x00--server\x00")
	l := linuxPlatform{procfs: d}
	if got := l.name(procInfo{pid: 7, name: ".ape-1.10"}); got != "opencoti-llamafile-0.10.5-c13-bare.llamafile" {
		t.Errorf("named %q", got)
	}
	if got := l.name(procInfo{pid: 8, name: "xollama"}); got != "xollama" {
		t.Errorf("a plain process renamed to %q", got)
	}
}
