package xollama

import (
	"strings"
	"testing"
)

func TestTheDefaultNoticesHideOneEngineStatusSpan(t *testing.T) {
	tmpl, err := ParseNoticeTemplates(DefaultNoticeTemplates)
	if err != nil {
		t.Fatal(err)
	}
	if len(tmpl.Think) != 4 || len(tmpl.Answer) != 4 || tmpl.Message == "" {
		t.Fatalf("the default ladder is not 4 + 4 + a message: %+v", tmpl)
	}
	spans := tmpl.Hide()
	if len(spans) != 1 || spans[0].Open != "<engine_status" || spans[0].Close == nil || *spans[0].Close != "</engine_status>" {
		t.Fatalf("hide = %+v, want one <engine_status ... </engine_status> span", spans)
	}
}

// The engine's rule (opencoti contract v2): '<' and '[' openings close with
// '/' after their first character; any other opening runs to the line's end.
func TestNoticeSpansFollowTheEnginesClosingRule(t *testing.T) {
	tmpl := NoticeTemplates{
		Think:   []NoticeStep{{Template: "[note budget={left}] wrap up [/note]"}},
		Answer:  []NoticeStep{{Template: "## {left} left"}, {Template: "[note again [/note]"}},
		Message: "<engine_status>waiting</engine_status>",
	}
	spans := tmpl.Hide()
	if len(spans) != 3 {
		t.Fatalf("hide = %+v, want three openings, each once", spans)
	}
	want := []struct{ open, close string }{{"[note", "[/note]"}, {"##", ""}, {"<engine_status", "</engine_status>"}}
	for i, w := range want {
		got := ""
		if spans[i].Close != nil {
			got = *spans[i].Close
		}
		if spans[i].Open != w.open || got != w.close {
			t.Errorf("span %d = %q..%q, want %q..%q", i, spans[i].Open, got, w.open, w.close)
		}
	}
}

func TestANoticeFileWithAnUnknownFieldIsRefused(t *testing.T) {
	_, err := ParseNoticeTemplates([]byte(`{"thinking": []}`))
	if err == nil || !strings.Contains(err.Error(), "thinking") {
		t.Fatalf("err = %v, want the unknown field named", err)
	}
}

func TestNoticesTemplatesTakeADigestOrNone(t *testing.T) {
	for _, v := range []string{"sha256:" + strings.Repeat("a", 64), NoticeTemplatesNone} {
		if err := (&Notices{Templates: v}).validate(); err != nil {
			t.Errorf("%q: %v", v, err)
		}
	}
	if err := (&Notices{Templates: "my-file.json"}).validate(); err == nil {
		t.Error("a file name was accepted as a digest")
	}
}
