package xollama

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// xollama-hook: notices — opencoti generation notices, contract v2.
//
// A model's notice templates are their own layer, MediaTypeNotices, holding
// exactly the JSON the engine's --notice-templates FILE takes:
// {"think"?: [steps], "answer"?: [steps], "message"?: "template"}. The launch
// passes the layer's blob path; a model without one gets DefaultNoticeTemplates,
// so xollama always knows what was planted and which spans a client hides
// (owner, 2026-10-15). The engine is the judge of a file: create checks it with
// the engine's --print-notice-config, and an engine refusing one at boot names
// the field in its last stderr line.

// MediaTypeNotices is the manifest media type of a model's notice templates.
const MediaTypeNotices = "application/vnd.xollama.notices"

// NoticeTemplatesNone is the notices.templates value that detaches a model's
// templates, so it gets the defaults again.
const NoticeTemplatesNone = "none"

// DefaultNoticeTemplates is xollama's notice ladder for a model that has none:
// opencoti's form F, four levels per phase at 60/70/80/90 %, measured on ten
// models (opencoti op-128).
//
//go:embed notices_default.json
var DefaultNoticeTemplates []byte

// Notices is the create request's word on a model's notice templates. It is
// never stored: the layer is the truth, and show reads the templates from it.
type Notices struct {
	// Templates is the sha256 digest of a templates file already in the model
	// store, or NoticeTemplatesNone to detach the model's. Empty leaves the
	// model's layer as it is.
	Templates string `json:"templates,omitempty"`
}

// IsZero reports whether the block states nothing.
func (n *Notices) IsZero() bool { return n == nil || n.Templates == "" }

func (n *Notices) validate() error {
	if n == nil || n.Templates == "" || n.Templates == NoticeTemplatesNone {
		return nil
	}
	if !draftHeadDigest.MatchString(n.Templates) {
		return fmt.Errorf("xollama config: notices.templates %q is neither a sha256 digest nor %q", n.Templates, NoticeTemplatesNone)
	}
	return nil
}

// NoticeStep is one level of a phase's ladder.
type NoticeStep struct {
	AtPercent *int   `json:"at_percent,omitempty"`
	Left      *int   `json:"left,omitempty"`
	Template  string `json:"template,omitempty"`
}

// NoticeTemplates is the --notice-templates file.
type NoticeTemplates struct {
	Think   []NoticeStep `json:"think,omitempty"`
	Answer  []NoticeStep `json:"answer,omitempty"`
	Message string       `json:"message,omitempty"`
}

// ParseNoticeTemplates reads a templates file the way the engine does in
// shape: a JSON object with no fields but think, answer and message. The
// engine's own check (create runs it) judges the templates themselves.
func ParseNoticeTemplates(data []byte) (NoticeTemplates, error) {
	var t NoticeTemplates
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&t); err != nil {
		return t, fmt.Errorf("notice templates: %w", err)
	}
	if dec.More() {
		return t, errors.New("notice templates: more than one JSON value")
	}
	return t, nil
}

// CompleteNoticeTemplates is a model's templates file with every part it
// leaves out taken from DefaultNoticeTemplates. The engine would fill a
// missing phase or message from its own built-in ladder, whose openings the
// model's hide spans would not list, and a client would then show those
// notices; a complete file is what the engine plants and the spans describe.
func CompleteNoticeTemplates(data []byte) ([]byte, error) {
	t, err := ParseNoticeTemplates(data)
	if err != nil {
		return nil, err
	}
	def, err := ParseNoticeTemplates(DefaultNoticeTemplates)
	if err != nil {
		return nil, err
	}
	if t.Think == nil {
		t.Think = def.Think
	}
	if t.Answer == nil {
		t.Answer = def.Answer
	}
	if t.Message == "" {
		t.Message = def.Message
	}
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(t); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// NoticeSpan is what a client hides: a notice starts at a line start with Open
// and runs through Close, or to the end of its line when Close is nil.
type NoticeSpan struct {
	Open  string  `json:"open"`
	Close *string `json:"close"`
}

// Hide is the span of every template in the file, in order, each opening
// once. It is the engine's rule (opencoti llamafile-flags.md, contract v2):
// the opening runs to the first blank, tab, '>', ']', ':' or line break; an
// opening with '<' or '[' closes with '/' after its first character, through
// the next '>' or ']'.
func (t NoticeTemplates) Hide() []NoticeSpan {
	var spans []NoticeSpan
	seen := map[string]bool{}
	add := func(tmpl string) {
		open := noticeOpening(tmpl)
		if open == "" || seen[open] {
			return
		}
		seen[open] = true
		spans = append(spans, NoticeSpan{Open: open, Close: noticeClose(tmpl, open)})
	}
	for _, s := range t.Think {
		add(s.Template)
	}
	for _, s := range t.Answer {
		add(s.Template)
	}
	add(t.Message)
	return spans
}

func noticeOpening(tmpl string) string {
	if i := strings.IndexAny(tmpl, " \t>]:\r\n"); i >= 0 {
		return tmpl[:i]
	}
	return tmpl
}

func noticeClose(tmpl, open string) *string {
	if len(open) < 2 || (open[0] != '<' && open[0] != '[') {
		return nil
	}
	stem := open[:1] + "/" + open[1:]
	i := strings.Index(tmpl, stem)
	if i < 0 {
		return nil
	}
	rest := tmpl[i:]
	j := strings.IndexAny(rest, ">]")
	if j < 0 {
		return nil
	}
	c := rest[:j+1]
	return &c
}
