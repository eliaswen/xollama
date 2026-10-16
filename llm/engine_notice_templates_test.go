package llm

import (
	"slices"
	"testing"
)

// The model's templates reach opencoti as --notice-templates, and nothing
// else carries the flag: stock llama.cpp and an engine without contract v2
// would refuse the whole command line for it.
func TestNoticeTemplatesReachOnlyAnEngineThatTakesThem(t *testing.T) {
	old := engineTakesNoticeTemplates
	t.Cleanup(func() { engineTakesNoticeTemplates = old })

	engineTakesNoticeTemplates = func() bool { return true }
	got := appendNoticeTemplatesArgs([]string{"-c", "8192"}, "/store/blobs/sha256-ab", true)
	if want := []string{"-c", "8192", "--notice-templates", "/store/blobs/sha256-ab"}; !slices.Equal(got, want) {
		t.Fatalf("opencoti with v2: %q, want %q", got, want)
	}
	if got := appendNoticeTemplatesArgs([]string{"-c", "8192"}, "/store/blobs/sha256-ab", false); slices.Contains(got, "--notice-templates") {
		t.Fatalf("stock llama.cpp got the flag: %q", got)
	}
	if got := appendNoticeTemplatesArgs([]string{"-c", "8192"}, "", true); slices.Contains(got, "--notice-templates") {
		t.Fatalf("no file, yet the flag: %q", got)
	}

	engineTakesNoticeTemplates = func() bool { return false }
	if got := appendNoticeTemplatesArgs([]string{"-c", "8192"}, "/store/blobs/sha256-ab", true); slices.Contains(got, "--notice-templates") {
		t.Fatalf("an engine without contract v2 got the flag: %q", got)
	}
}
