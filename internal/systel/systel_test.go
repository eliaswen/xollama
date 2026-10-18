package systel

import (
	"sync"
	"testing"
	"time"
)

type countingReader struct {
	mu     sync.Mutex
	reads  int
	resets int
}

func (c *countingReader) read(s *Snapshot) {
	c.mu.Lock()
	c.reads++
	c.mu.Unlock()
}

func (c *countingReader) reset() {
	c.mu.Lock()
	c.resets++
	c.mu.Unlock()
}

func (c *countingReader) counts() (int, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.reads, c.resets
}

// Any number of polls between two refreshes cost nothing: they get the
// cached copy.
func TestPollsReadTheCachedSnapshot(t *testing.T) {
	r := &countingReader{}
	s := newSampler(time.Hour, time.Hour, []reader{r}, time.Now)
	defer s.Close()
	first := s.Get()
	for range 1000 {
		if got := s.Get(); !got.At.Equal(first.At) {
			t.Fatal("a poll took a new reading")
		}
	}
	if reads, _ := r.counts(); reads != 1 {
		t.Fatalf("1000 polls made %d readings, want 1", reads)
	}
}

// Nobody polling, nothing read; the next poll forgets the old counters, so
// no rate spans the idle gap.
func TestRefreshingStopsWhenNobodyPolls(t *testing.T) {
	var mu sync.Mutex
	now := time.Unix(0, 0)
	clock := func() time.Time { mu.Lock(); defer mu.Unlock(); return now }
	advance := func(d time.Duration) { mu.Lock(); now = now.Add(d); mu.Unlock() }

	r := &countingReader{}
	s := newSampler(5*time.Millisecond, time.Minute, []reader{r}, clock)
	defer s.Close()
	s.Get()
	deadline := time.Now().Add(5 * time.Second)
	for reads, _ := r.counts(); reads < 3; reads, _ = r.counts() {
		if time.Now().After(deadline) {
			t.Fatal("the timer did not refresh while polled")
		}
		time.Sleep(time.Millisecond)
	}
	advance(2 * time.Minute) // past idle
	for {
		s.mu.Lock()
		running := s.running
		s.mu.Unlock()
		if !running {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the timer kept reading with nobody polling")
		}
		time.Sleep(time.Millisecond)
	}
	settled, _ := r.counts()
	time.Sleep(30 * time.Millisecond)
	if reads, _ := r.counts(); reads != settled {
		t.Fatalf("read %d times after it stopped", reads-settled)
	}
	s.Get()
	if _, resets := r.counts(); resets != 2 {
		t.Fatalf("resets %d, want one per start", resets)
	}
}

func TestPct(t *testing.T) {
	if pct(0, 1, 0, 2, false) != nil {
		t.Error("no earlier reading, no rate")
	}
	if pct(1, 2, 4, 4, true) != nil {
		t.Error("nothing moved on the denominator, no rate")
	}
	if v := pct(10, 30, 100, 140, true); v == nil || *v != 50 {
		t.Errorf("got %v", v)
	}
}

// A reused process id is not a child: a process that started before the one
// it names as parent was the child of someone else.
func TestTheTreeSkipsAReusedParentID(t *testing.T) {
	all := []procInfo{
		{pid: 10, ppid: 1, start: 100},
		{pid: 11, ppid: 10, start: 150},
		{pid: 12, ppid: 10, start: 50}, // older than 10: its parent died, 10 was reused
		{pid: 13, ppid: 11, start: 160},
		{pid: 14, ppid: 2, start: 170},
	}
	got := subtree(all, 10)
	var pids []int
	for _, p := range got {
		pids = append(pids, p.pid)
	}
	if len(pids) != 3 || pids[0] != 10 || pids[1] != 11 || pids[2] != 13 {
		t.Fatalf("tree %v, want [10 11 13]", pids)
	}
}

func TestNormalizePCI(t *testing.T) {
	for in, want := range map[string]string{
		"00000000:01:00.0": "0000:01:00.0",
		"0000:E3:00.0":     "0000:e3:00.0",
		"01:00.0":          "0000:01:00.0",
	} {
		if got := normalizePCI(in); got != want {
			t.Errorf("%s -> %s, want %s", in, got, want)
		}
	}
}
