package llm

import "testing"

// The body as opencoti c11 r3 writes it (0618 gate, draft-auto on a held 3090).
const droppedProps = `{"features":["polykv_v1","drafter_state_v1"],"opencoti":{"speculative":{"types":"none","drafter":{"state":"dropped","kind":"assistant","reason":"fit_shortfall","shortfall_mib":51,"head_mib":486,"freed_mib":614,"decided_at":"fit"}}}}`

func TestTheEnginesDrafterRecordIsRead(t *testing.T) {
	d := parseDrafterState([]byte(droppedProps))
	if d == nil || d.State != "dropped" || d.Kind != "assistant" || d.Reason != "fit_shortfall" {
		t.Fatalf("record = %+v", d)
	}
	if d.ShortfallMiB == nil || *d.ShortfallMiB != 51 || d.HeadMiB == nil || *d.HeadMiB != 486 || d.FreedMiB == nil || *d.FreedMiB != 614 {
		t.Fatalf("MiB = %v %v %v", d.ShortfallMiB, d.HeadMiB, d.FreedMiB)
	}

	none := parseDrafterState([]byte(`{"features":["drafter_state_v1"],"opencoti":{"speculative":{"drafter":{"state":"none","kind":null,"reason":null,"shortfall_mib":null}}}}`))
	if none == nil || none.State != "none" || none.Kind != "" || none.ShortfallMiB != nil {
		t.Fatalf("none = %+v", none)
	}
}

func TestADrafterStateNewerThanThisBuildIsShownAsDropped(t *testing.T) {
	d := parseDrafterState([]byte(`{"features":["drafter_state_v1"],"opencoti":{"speculative":{"drafter":{"state":"parked"}}}}`))
	if d == nil || d.State != "dropped" {
		t.Fatalf("an unknown state = %+v, want dropped", d)
	}
}

func TestNoRecordWithoutTheFeature(t *testing.T) {
	for _, body := range []string{
		`{"features":["polykv_v1"],"opencoti":{"speculative":{"drafter":{"state":"dropped"}}}}`,
		`{"default_generation_settings":{}}`,
		`not json`,
	} {
		if d := parseDrafterState([]byte(body)); d != nil {
			t.Fatalf("%s: record = %+v, want none", body, d)
		}
	}
}
