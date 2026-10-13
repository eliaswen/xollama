package server

import (
	"errors"
	"fmt"
	"strings"

	"golang.org/x/mod/semver"

	"github.com/ollama/ollama/version"
)

// errRequires is a model that names an xOllama release newer than this one.
var errRequires = errors.New("model requires a newer xollama")

// forkRequirement reports whether a REQUIRES value names an xOllama release
// (0.40.3-xollama.3, 0.40.3-xollama, 0.41.0-rc.1.xollama) rather than an
// upstream version, and returns it in the form semver compares.
func forkRequirement(requires string) (string, bool) {
	v := "v" + strings.TrimPrefix(requires, "v")
	if !semver.IsValid(v) || !strings.Contains(semver.Prerelease(v), "xollama") {
		return "", false
	}
	return v, true
}

// checkForkRequires answers for a requirement that names an xOllama release.
//
// Upstream drops the running version's suffix before it compares, so that a
// release candidate can test its release's models. Every xOllama version
// lives in that suffix, and a re-release keeps upstream's number, so on that
// rule 0.40.3-xollama.2 passes a model that needs 0.40.3-xollama.3. Here the
// whole version is compared: plain semver already orders a candidate, the
// release and its re-releases, and a later upstream number beats any of them.
// A build with no version (0.0.0) passes, as it does upstream.
func checkForkRequires(requires, running string) (handled bool, err error) {
	want, ok := forkRequirement(requires)
	if !ok {
		return false, nil
	}
	if running == "0.0.0" {
		return true, nil
	}
	have := "v" + strings.TrimPrefix(running, "v")
	if !semver.IsValid(have) {
		return true, nil
	}
	if semver.Compare(want, have) > 0 {
		return true, fmt.Errorf("%w: %s or later (this is %s)", errRequires, want, have)
	}
	return true, nil
}

// loadRequires refuses a load of a model that names a newer xOllama release.
// A pull already refuses it; this is for the model that arrived another way
// (created here, copied in, or pulled by a later build than the one running),
// which would otherwise fail inside the engine with an error naming a tensor.
// An upstream-form requirement stays a pull-time check, as upstream has it.
func loadRequires(m *Model) error {
	if m == nil {
		return nil
	}
	if _, err := checkForkRequires(m.Config.Requires, version.Version); err != nil {
		return fmt.Errorf("%s: %w", m.ShortName, err)
	}
	return nil
}
