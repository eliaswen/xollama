package parsers

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

// feedGemma4Reopened runs chunks through a thinking-enabled parser and returns
// what came out of each channel.
func feedGemma4Reopened(t *testing.T, chunks []string) (string, string, []api.ToolCall) {
	t.Helper()
	parser := &Gemma4Parser{hasThinkingSupport: true}
	parser.Init(nil, nil, &api.ThinkValue{Value: true})
	var content, thinking strings.Builder
	var calls []api.ToolCall
	for i, chunk := range chunks {
		c, th, tc, err := parser.Add(chunk, i == len(chunks)-1)
		if err != nil {
			t.Fatalf("Add() error on chunk %d: %v", i, err)
		}
		content.WriteString(c)
		thinking.WriteString(th)
		calls = append(calls, tc...)
	}
	return content.String(), thinking.String(), calls
}

// splitRunes streams s one rune at a time, the worst chunking a server sends.
func splitRunes(s string) []string {
	var out []string
	for _, r := range s {
		out = append(out, string(r))
	}
	return out
}

func TestGemma4Parser_ReopenedChannelInsideThinking(t *testing.T) {
	cases := map[string]string{
		"thought":    "<|channel>thought\nLet's try to restore first.\n<|channel>thought\nThe file is back.<channel|>Done.",
		"angle name": "<|channel>thought\nLet's try to restore first.\n\n<|channel><thought>\nThe file is back.<channel|>Done.",
	}
	for name, stream := range cases {
		for _, chunking := range []string{"whole", "runes"} {
			t.Run(name+"/"+chunking, func(t *testing.T) {
				chunks := []string{stream}
				if chunking == "runes" {
					chunks = splitRunes(stream)
				}
				content, thinking, calls := feedGemma4Reopened(t, chunks)
				for _, leak := range []string{"<|channel>", "thought\n", "<thought>"} {
					if strings.Contains(thinking, leak) {
						t.Errorf("%q leaked into thinking: %q", leak, thinking)
					}
				}
				if !strings.Contains(thinking, "restore first.") || !strings.Contains(thinking, "The file is back.") {
					t.Errorf("thinking lost text: %q", thinking)
				}
				if content != "Done." {
					t.Errorf("content = %q, want %q", content, "Done.")
				}
				if len(calls) != 0 {
					t.Errorf("unexpected calls: %+v", calls)
				}
			})
		}
	}
}

func TestGemma4Parser_ReopenedChannelAtEndOfStream(t *testing.T) {
	for _, chunking := range []string{"whole", "runes"} {
		stream := "<|channel>thought\nLet's do it.\n<|channel>thought"
		chunks := []string{stream}
		if chunking == "runes" {
			chunks = splitRunes(stream)
		}
		_, thinking, _ := feedGemma4Reopened(t, chunks)
		if thinking != "Let's do it." {
			t.Errorf("%s: thinking = %q, want %q", chunking, thinking, "Let's do it.")
		}
	}
}

func TestGemma4Parser_ToolCallAfterReopenedChannel(t *testing.T) {
	stream := "<|channel>thought\nLet's do it.\n<|channel>thought\n" + `<|tool_call>call:editor{path:<|"|>manic_miner.html<|"|>}<tool_call|>`
	want := []api.ToolCall{{
		Function: api.ToolCallFunction{
			Name:      "editor",
			Arguments: testArgs(map[string]any{"path": "manic_miner.html"}),
		},
	}}
	for _, chunking := range []string{"whole", "runes"} {
		chunks := []string{stream}
		if chunking == "runes" {
			chunks = splitRunes(stream)
		}
		content, thinking, calls := feedGemma4Reopened(t, chunks)
		if diff := cmp.Diff(want, calls, argsComparer); diff != "" {
			t.Errorf("%s: tool calls mismatch (-want +got):\n%s", chunking, diff)
		}
		if strings.Contains(thinking, "<|channel>") || strings.Contains(thinking, "tool_call") {
			t.Errorf("%s: leaked into thinking: %q", chunking, thinking)
		}
		if content != "" {
			t.Errorf("%s: content = %q, want none", chunking, content)
		}
	}
}
