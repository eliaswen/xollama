package parsers

import "strings"

// CallProgress follows the raw text of a response and reports what the model
// has written of a tool call so far, while the parser itself is still holding
// the call back until it is complete. It reads nothing but the parser's own
// delimiters (ToolCallTagger), so it needs no change in a parser and works for
// every one that names them.
type CallProgress struct {
	start, end string
	held       string // text not yet attributed: it may end in part of a tag
	in         bool
	index      int // of the call being written, from 0
	tokens     int // deltas seen since the call opened
}

// CallDelta is one step of a call being written.
type CallDelta struct {
	Index  int    // which call of the response
	Text   string // raw text added, in the model's own syntax, tags excluded
	Tokens int    // deltas of the response this call has taken so far
	Done   bool   // the closing delimiter was read
}

// NewCallProgress returns a follower for p, or nil when p does not name the
// delimiters of its tool calls. A nil follower reports nothing.
func NewCallProgress(p Parser) *CallProgress {
	if p == nil || !p.HasToolSupport() {
		return nil
	}
	tagger, ok := p.(ToolCallTagger)
	if !ok {
		return nil
	}
	start, end := tagger.ToolCallTags()
	if start == "" {
		return nil
	}
	return &CallProgress{start: start, end: end, index: -1}
}

// Add takes the next piece of raw response text and returns the steps it
// holds: usually none or one, two when a call closes and the next opens in
// the same piece. Text never contains a delimiter, whole or in part.
func (c *CallProgress) Add(s string) []CallDelta {
	if c == nil || s == "" {
		return nil
	}
	c.held += s
	if c.in {
		c.tokens++
	}

	var out []CallDelta
	for {
		if !c.in {
			idx := strings.Index(c.held, c.start)
			if idx < 0 {
				c.held = c.held[len(c.held)-tagOverlap(c.held, c.start):]
				return out
			}
			c.held = c.held[idx+len(c.start):]
			c.in = true
			c.index++
			c.tokens = 1
		}

		if c.end != "" {
			if idx := strings.Index(c.held, c.end); idx >= 0 {
				out = append(out, CallDelta{Index: c.index, Text: c.held[:idx], Tokens: c.tokens, Done: true})
				c.held = c.held[idx+len(c.end):]
				c.in = false
				continue
			}
		}

		keep := 0
		if c.end != "" {
			keep = tagOverlap(c.held, c.end)
		}
		text := c.held[:len(c.held)-keep]
		c.held = c.held[len(c.held)-keep:]
		// An opened call is reported even before its first character, so a
		// client learns of it at once.
		if text != "" || c.tokens == 1 {
			out = append(out, CallDelta{Index: c.index, Text: text, Tokens: c.tokens})
		}
		return out
	}
}

// tagOverlap is the length of the longest suffix of s that begins tag without
// being all of it.
func tagOverlap(s, tag string) int {
	n := min(len(tag)-1, len(s))
	for ; n > 0; n-- {
		if strings.HasSuffix(s, tag[:n]) {
			return n
		}
	}
	return 0
}
