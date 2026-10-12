package tweak

import (
	"os"
	"strings"
	"testing"
)

const referencePage = "../../docs/xollama/tweak-reference.mdx"

// The reference page is the settings table, rendered. A setting added to the
// table without regenerating the page fails here, so no flag ships
// undocumented. Regenerate with:
//
//	XOLLAMA_UPDATE_DOCS=1 go test ./cmd/tweak -run TestTheFlagReferenceIsCurrent
func TestTheFlagReferenceIsCurrent(t *testing.T) {
	want := Reference()
	if os.Getenv("XOLLAMA_UPDATE_DOCS") != "" {
		if err := os.WriteFile(referencePage, []byte(want), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got, err := os.ReadFile(referencePage)
	if err != nil {
		t.Fatalf("%v (generate it: XOLLAMA_UPDATE_DOCS=1 go test ./cmd/tweak -run TestTheFlagReferenceIsCurrent)", err)
	}
	if strings.ReplaceAll(string(got), "\r\n", "\n") != want {
		t.Fatalf("%s is not what cmd/tweak generates: a setting was added, renamed or reworded.\nRegenerate it: XOLLAMA_UPDATE_DOCS=1 go test ./cmd/tweak -run TestTheFlagReferenceIsCurrent", referencePage)
	}
}

// Every flag the three commands take is on a page a reader can find: a
// setting in the generated reference, the rest in the guide.
func TestEveryTweakFlagIsDocumented(t *testing.T) {
	ref := Reference()
	guide, err := os.ReadFile("../../docs/xollama/tweak.mdx")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range fields {
		if !strings.Contains(ref, "`--"+f.name+"`") {
			t.Errorf("--%s is missing from the reference", f.name)
		}
	}
	for _, name := range []string{"--api-key", "--backend", "--disable", "--enable", "--link", "--priority", "--split", "--split-mode", "--clear", "--dry-run", "--json"} {
		if !strings.Contains(string(guide), "`"+name) {
			t.Errorf("%s (tweak server / tweak server gpu) is not in docs/xollama/tweak.mdx", name)
		}
	}
}
