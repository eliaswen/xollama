package parsers

import (
	"strings"
	"testing"
)

func feed(c *CallProgress, pieces ...string) []CallDelta {
	var all []CallDelta
	for _, p := range pieces {
		all = append(all, c.Add(p)...)
	}
	return all
}

func joined(ds []CallDelta, index int) string {
	var b strings.Builder
	for _, d := range ds {
		if d.Index == index {
			b.WriteString(d.Text)
		}
	}
	return b.String()
}

func TestCallProgressReportsACallWhileItIsWritten(t *testing.T) {
	c := NewCallProgress(ParserForName("gemma4"))
	if c == nil {
		t.Fatal("gemma4 names its tool-call tags; a follower was expected")
	}

	ds := feed(c, "Let me look.", "<|tool_call>", "call:edit{intent:<|\"|>fix", " the loop<|\"|>,", "path:<|\"|>a.go<|\"|>}", "<tool_call|>", " done")
	if got, want := joined(ds, 0), "call:edit{intent:<|\"|>fix the loop<|\"|>,path:<|\"|>a.go<|\"|>}"; got != want {
		t.Fatalf("text = %q, want %q", got, want)
	}
	if ds[0].Text != "" || ds[0].Tokens != 1 {
		t.Fatalf("the opening tag alone must report the call at once: %+v", ds[0])
	}
	last := ds[len(ds)-1]
	if !last.Done {
		t.Fatalf("the closing tag must end the call: %+v", last)
	}
	if last.Tokens != 5 {
		t.Fatalf("tokens = %d, want 5 (the pieces from the opening tag to the closing one)", last.Tokens)
	}
	for _, d := range ds {
		if strings.Contains(d.Text, "tool_call") {
			t.Fatalf("a delimiter leaked into the text: %q", d.Text)
		}
	}
}

func TestCallProgressHoldsBackAPartialTag(t *testing.T) {
	c := NewCallProgress(ParserForName("gemma4"))

	// The opening tag split over three pieces, the closing one over two.
	ds := feed(c, "text <|to", "ol_ca", "ll>call:x{a:1}<tool", "_call|>")
	if got, want := joined(ds, 0), "call:x{a:1}"; got != want {
		t.Fatalf("text = %q, want %q", got, want)
	}
	if !ds[len(ds)-1].Done {
		t.Fatal("the split closing tag must still end the call")
	}

	// Text that only looks like the start of the closing tag is released
	// once it turns out not to be one.
	c = NewCallProgress(ParserForName("gemma4"))
	ds = feed(c, "<|tool_call>a<tool", "bar>b")
	if got, want := joined(ds, 0), "a<toolbar>b"; got != want {
		t.Fatalf("text = %q, want %q", got, want)
	}
}

func TestCallProgressNumbersCalls(t *testing.T) {
	c := NewCallProgress(ParserForName("gemma4"))

	ds := feed(c, "<|tool_call>one<tool_call|><|tool_call>two", "<tool_call|>")
	if joined(ds, 0) != "one" || joined(ds, 1) != "two" {
		t.Fatalf("calls = %q and %q, want one and two (%+v)", joined(ds, 0), joined(ds, 1), ds)
	}
}

func TestCallProgressIsSilentOutsideACall(t *testing.T) {
	c := NewCallProgress(ParserForName("gemma4"))
	if ds := feed(c, "plain ", "text with < and | in it", ""); len(ds) != 0 {
		t.Fatalf("no call was opened, got %+v", ds)
	}
}

func TestCallProgressNeedsAParserThatNamesItsTags(t *testing.T) {
	// qwen3-coder parses tool calls but does not name its delimiters.
	if c := NewCallProgress(ParserForName("qwen3-coder")); c != nil {
		t.Fatal("a parser without ToolCallTags must give no follower")
	}
	var c *CallProgress
	if ds := c.Add("<|tool_call>x"); ds != nil {
		t.Fatalf("a nil follower reports nothing, got %+v", ds)
	}
}
