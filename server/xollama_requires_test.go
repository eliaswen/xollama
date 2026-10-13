package server

import (
	"errors"
	"strings"
	"testing"

	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
)

func TestARequirementCanNameAnXollamaRelease(t *testing.T) {
	tests := []struct {
		name     string
		requires string
		running  string
		refused  bool
	}{
		{"the re-release before it", "0.40.3-xollama.3", "0.40.3-xollama.2", true},
		{"the release itself", "0.40.3-xollama.3", "0.40.3-xollama.3", false},
		{"a later re-release", "0.40.3-xollama.3", "0.40.3-xollama.4", false},
		{"re-releases order by number, not by text", "0.40.3-xollama.9", "0.40.3-xollama.10", false},
		{"the first release is older than a re-release", "0.40.3-xollama.1", "0.40.3-xollama", true},
		{"the first release, required", "0.40.3-xollama", "0.40.3-xollama", false},
		{"a candidate is older than its release", "0.40.3-xollama", "0.40.3-rc.2.xollama", true},
		{"a candidate, required of a later candidate", "0.41.0-rc.1.xollama", "0.41.0-rc.2.xollama", false},
		{"a later upstream base", "0.40.3-xollama.3", "0.40.4-xollama", false},
		{"a later base's candidate", "0.40.3-xollama.3", "0.41.0-rc.1.xollama", false},
		{"an earlier upstream base", "0.40.3-xollama", "0.40.2-xollama.5", true},
		{"a build made after the release", "0.40.3-xollama.3", "0.40.3-xollama.2-5-g9d4e9d3", false},
		{"a build with no version", "0.40.3-xollama.3", "0.0.0", false},
		{"a v prefix", "v0.40.3-xollama.3", "0.40.3-xollama.2", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := checkPullRequires(tt.requires, tt.running)
			if tt.refused != (err != nil) {
				t.Fatalf("checkPullRequires(%q, %q) = %v, refused want %v", tt.requires, tt.running, err, tt.refused)
			}
			if err == nil {
				return
			}
			if !errors.Is(err, errRequires) {
				t.Fatalf("error %v is not errRequires", err)
			}
			for _, want := range []string{strings.TrimPrefix(tt.requires, "v"), tt.running} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not name %q", err, want)
				}
			}
		})
	}
}

// An upstream requirement keeps upstream's rule, on the fork's versions too:
// the running version's suffix is dropped before the comparison.
func TestAnUpstreamRequirementKeepsUpstreamsRule(t *testing.T) {
	tests := []struct {
		requires string
		running  string
		refused  bool
	}{
		{"0.40.3", "0.40.3-xollama.2", false},
		{"0.40.3", "0.40.3-rc.1.xollama", false},
		{"0.40.4", "0.40.3-xollama.2", true},
		{"0.40.3-rc1", "0.40.3-xollama.2", false},
	}
	for _, tt := range tests {
		if handled, _ := checkForkRequires(tt.requires, tt.running); handled {
			t.Errorf("checkForkRequires(%q) answered for an upstream requirement", tt.requires)
		}
		err := checkPullRequires(tt.requires, tt.running)
		if tt.refused != (err != nil) {
			t.Errorf("checkPullRequires(%q, %q) = %v, refused want %v", tt.requires, tt.running, err, tt.refused)
		}
		if err != nil && errors.Is(err, errRequires) {
			t.Errorf("an upstream requirement was refused with the fork's error: %v", err)
		}
	}
}

func TestALoadRefusesAModelThatNamesANewerXollama(t *testing.T) {
	old := version.Version
	version.Version = "0.40.3-xollama.2"
	t.Cleanup(func() { version.Version = old })

	m := &Model{ShortName: "whistle:vad", Config: model.ConfigV2{Requires: "0.40.3-xollama.3"}}
	err := loadRequires(m)
	if !errors.Is(err, errRequires) {
		t.Fatalf("loadRequires = %v, want errRequires", err)
	}
	if !strings.Contains(err.Error(), "whistle:vad") {
		t.Errorf("error %q does not name the model", err)
	}

	// Upstream checks its own form at pull only; a load does not start to.
	m.Config.Requires = "0.99.0"
	if err := loadRequires(m); err != nil {
		t.Errorf("an upstream requirement was checked at load: %v", err)
	}
	m.Config.Requires = "0.40.3-xollama.2"
	if err := loadRequires(m); err != nil {
		t.Errorf("a requirement this build meets was refused: %v", err)
	}
}
