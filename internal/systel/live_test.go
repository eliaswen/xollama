package systel

import (
	"encoding/json"
	"os"
	"testing"
	"time"
)

// TestLiveProbe reads this machine twice, 1.5 s apart, and writes both
// snapshots as a.json and b.json into the directory SYSTEL_LIVE names. It is
// how a new platform is checked by hand; without SYSTEL_LIVE it is skipped.
func TestLiveProbe(t *testing.T) {
	if os.Getenv("SYSTEL_LIVE") == "" {
		t.Skip()
	}
	s := New(time.Second, 10*time.Second)
	defer s.Close()
	t0 := time.Now()
	a := s.Get()
	t.Logf("first read %v", time.Since(t0))
	time.Sleep(1500 * time.Millisecond)
	t1 := time.Now()
	b := s.Get()
	t.Logf("cached read %v", time.Since(t1))
	ja, _ := json.MarshalIndent(a, "", " ")
	jb, _ := json.MarshalIndent(b, "", " ")
	dir := os.Getenv("SYSTEL_LIVE")
	os.WriteFile(dir+"/a.json", ja, 0o644)
	os.WriteFile(dir+"/b.json", jb, 0o644)
}
