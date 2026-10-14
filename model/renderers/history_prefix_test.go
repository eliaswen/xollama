package renderers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// A tool loop's next request must begin with the previous request's prompt
// followed by exactly what the model generated. Where it does not, the cache
// re-reads from the divergence; on a recurrent model (Qwen3.5 and 3.6 are
// Gated Delta Net hybrids) that is the whole previous step. Measured through
// xOllama with thinking off on qwen3.5:2b: agent step 3 reused 770 of 1254
// prompt tokens before this, because the history dropped the empty think
// block the generation prompt had opened the turn with.

func historyTool() []api.Tool {
	return []api.Tool{{Type: "function", Function: api.ToolFunction{Name: "write_file", Description: "write", Parameters: api.ToolFunctionParameters{Type: "object"}}}}
}

func historyCall(path string) api.ToolCall {
	args := api.NewToolCallFunctionArguments()
	args.Set("path", path)
	return api.ToolCall{Function: api.ToolCallFunction{Name: "write_file", Arguments: args}}
}

// generated is what a Qwen3.5-family model writes after the generation prompt.
func qwenGenerated(think bool, content string, call api.ToolCall) string {
	var sb strings.Builder
	if think {
		sb.WriteString("plan\n</think>\n\n")
	}
	sb.WriteString(content)
	if content != "" {
		sb.WriteString("\n\n")
	}
	sb.WriteString("<tool_call>\n<function=" + call.Function.Name + ">\n")
	for name, value := range call.Function.Arguments.All() {
		sb.WriteString("<parameter=" + name + ">\n" + formatToolCallArgument(value) + "\n</parameter>\n")
	}
	sb.WriteString("</function>\n</tool_call>")
	return sb.String()
}

func assistantStep(think bool, content string, call api.ToolCall) api.Message {
	m := api.Message{Role: "assistant", Content: content, ToolCalls: []api.ToolCall{call}}
	if think {
		m.Thinking = "plan"
	}
	return m
}

func TestQwenToolLoopHistoryKeepsTheGeneratedPrefix(t *testing.T) {
	for _, name := range []string{"qwen3.5", "tev1", "qwen3.8", "ornith"} {
		for _, think := range []bool{false, true} {
			for _, content := range []string{"", "Writing it."} {
				tv := &api.ThinkValue{Value: think}
				user := api.Message{Role: "user", Content: "write two files"}
				c1, c2 := historyCall("a.txt"), historyCall("b.txt")
				a1, a2 := assistantStep(think, content, c1), assistantStep(think, content, c2)
				t1 := api.Message{Role: "tool", Content: "ok", ToolName: "write_file"}
				t2 := api.Message{Role: "tool", Content: "ok", ToolName: "write_file"}

				steps := [][]api.Message{{user}, {user, a1, t1}, {user, a1, t1, a2, t2}}
				outputs := []string{qwenGenerated(think, content, c1), qwenGenerated(think, content, c2)}
				for s := range 2 {
					prev, err := RenderWithRenderer(name, steps[s], historyTool(), tv)
					if err != nil {
						t.Fatal(err)
					}
					next, err := RenderWithRenderer(name, steps[s+1], historyTool(), tv)
					if err != nil {
						t.Fatal(err)
					}
					want := prev + outputs[s]
					if !strings.HasPrefix(next, want) {
						t.Errorf("%s think=%v content=%q step %d: the next prompt does not begin with the generated one\n want prefix …%q\n got        …%q",
							name, think, content, s+2, want[max(0, len(want)-120):], next[max(0, len(want)-120):min(len(next), len(want)+40)])
					}
				}
			}
		}
	}
}

func TestGemma4LargeToolLoopHistoryKeepsTheGeneratedPrefix(t *testing.T) {
	r := &Gemma4Renderer{emptyBlockOnNothink: true}
	for _, think := range []bool{false, true} {
		tv := &api.ThinkValue{Value: think}
		user := api.Message{Role: "user", Content: "write two files"}
		c1, c2 := historyCall("a.txt"), historyCall("b.txt")
		a1, a2 := assistantStep(think, "", c1), assistantStep(think, "", c2)
		t1 := api.Message{Role: "tool", Content: "ok", ToolName: "write_file"}
		t2 := api.Message{Role: "tool", Content: "ok", ToolName: "write_file"}
		steps := [][]api.Message{{user}, {user, a1, t1}, {user, a1, t1, a2, t2}}
		calls := []api.ToolCall{c1, c2}
		for s := range 2 {
			prev, err := RenderWithRenderer("gemma4-large", steps[s], historyTool(), tv)
			if err != nil {
				t.Fatal(err)
			}
			next, err := RenderWithRenderer("gemma4-large", steps[s+1], historyTool(), tv)
			if err != nil {
				t.Fatal(err)
			}
			out := r.formatToolCall(calls[s])
			if think {
				if strings.HasSuffix(prev, "<|channel>thought\n") {
					out = "plan\n<channel|>" + out
				} else {
					out = "<|channel>thought\nplan\n<channel|>" + out
				}
			}
			want := prev + out
			if !strings.HasPrefix(next, want) {
				t.Errorf("gemma4-large think=%v step %d: the next prompt does not begin with the generated one\n want prefix …%q\n got        …%q",
					think, s+2, want[max(0, len(want)-120):], next[max(0, len(want)-120):min(len(next), len(want)+40)])
			}
		}
	}
}

func TestQwen35TurnsBeforeTheLastUserKeepTheTemplateForm(t *testing.T) {
	// One re-read per user turn is the template's own design: an answer
	// before the last user message is rendered without a think block, with
	// thinking off as on. Only turns after it are rendered as generated.
	for _, think := range []bool{false, true} {
		got, err := RenderWithRenderer("qwen3.5", []api.Message{
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "hello", Thinking: "greet"},
			{Role: "user", Content: "again"},
		}, nil, &api.ThinkValue{Value: think})
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(got, "<|im_start|>assistant\nhello<|im_end|>") {
			t.Errorf("think=%v: an earlier answer is no longer in the template's form:\n%q", think, got)
		}
	}
}
