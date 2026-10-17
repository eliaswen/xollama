package engine

import "testing"

// An operator's engine is asked with --print-notice-config: v3 is its block's
// "v3" member, the copy bar that member's copy_bar (opencoti mails #1576-#1589).
func TestAPrintedBlockSaysWhetherTheEngineTakesV3AndTheCopyBar(t *testing.T) {
	for out, want := range map[string][2]bool{
		`{"version":2,"hide":[]}`:                                                {false, false},
		`{"version":2,"v3":{"think":[],"answer":[]}}`:                            {true, false},
		`{"v3":{"copy_bar":{"think":true,"answer":true}}}`:                       {true, true},
		`{"generation_notice":{"v3":{"copy_bar":{"think":true,"answer":true}}}}`: {true, true},
		`{"generation_notice":{"version":2}}`:                                    {false, false},
		`not json`:                                                               {false, false},
	} {
		v3, bar := printedV3([]byte(out))
		if [2]bool{v3, bar} != want {
			t.Errorf("%s: v3 %v copy_bar %v, want %v", out, v3, bar, want)
		}
	}
}
