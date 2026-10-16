package parsers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestAPathsBackslashInAnArrayParameterStaysABackslash(t *testing.T) {
	array := api.PropertyType{"array"}
	cases := []struct {
		name string
		raw  string
		want any
	}{
		// pandorum 2026-10-15: Cerebriline's run_commands, as the model wrote it.
		{"relative exe", `[".\run_game.exe manic_miner.html"]`, []any{`.\run_game.exe manic_miner.html`}},
		{"parent dir", `["..\tools\build.cmd"]`, []any{`..\tools\build.cmd`}},
		{"drive", `["dir C:\tmp\new\results"]`, []any{`dir C:\tmp\new\results`}},
		{"after an escaped separator", `["type C:\\Users\\manni\new.txt"]`, []any{`type C:\Users\manni\new.txt`}},
		{"quoted path", `["cd \"C:\temp\" && .\run.bat"]`, []any{`cd "C:\temp" && .\run.bat`}},
		{"mixed in one call", `[".\run_game.exe x --dir C:\\tmp\\new"]`, []any{`.\run_game.exe x --dir C:\tmp\new`}},
		// Escapes that are prose or code stay what JSON says.
		{"prose newline", `["done.\nNext step"]`, []any{"done.\nNext step"}},
		{"list newline", `["- [x] find manic_miner.html\n- [ ] run it"]`, []any{"- [x] find manic_miner.html\n- [ ] run it"}},
		{"label colon", `["ratio:\tvalue"]`, []any{"ratio:\tvalue"}},
		{"code tab", `["a\tb", "x\ny"]`, []any{"a\tb", "x\ny"}},
		{"escaped backslash only", `["C:\\Users\\manni"]`, []any{`C:\Users\manni`}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := parseTypedToolValue(tc.raw, array)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("parseTypedToolValue(%s) mismatch (-want +got):\n%s", tc.raw, diff)
			}
		})
	}
}

func TestAPathsBackslashInAnObjectParameterStaysABackslash(t *testing.T) {
	got := parseTypedToolValue(`{"cwd": ".\build", "note": "ok.\nnext"}`, api.PropertyType{"object"})
	want := map[string]any{"cwd": `.\build`, "note": "ok.\nnext"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("mismatch (-want +got):\n%s", diff)
	}
}
