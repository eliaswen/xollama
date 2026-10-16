package tweak

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/xollama"
)

// xollama-hook: notices — see docs/xollama/notices.mdx.

func noticeFields() []field {
	return []field{{
		name:  "notice-templates",
		path:  "notices.templates",
		title: "Notice templates — what the engine plants when room runs low",
		help: "Generation notices tell the model, inside its own output, how much room is\n" +
			"left (`{left}` of `{total}` tokens) for thinking and for the answer, and that\n" +
			"a message is waiting. This attaches a JSON file of your own wording, in the\n" +
			"engine's --notice-templates shape: {\"think\": [steps], \"answer\": [steps],\n" +
			"\"message\": \"template\"}, each step {\"at_percent\": 1..99 | \"left\": N,\n" +
			"\"template\": \"...\"}. The engine checks the file before it is attached and\n" +
			"names the field it refuses. A part the file leaves out is xollama's default.\n" +
			"`none` goes back to xollama's defaults: four levels per phase in an\n" +
			"<engine_status> tag, measured on ten models.",
		kind:      kindBlob,
		modelOnly: true,
		get: func(c *xollama.Config) string {
			return orEmpty(c.Notices != nil, func() string { return c.Notices.Templates })
		},
		set: setNoticeTemplates,
	}}
}

func setNoticeTemplates(c *xollama.Config, v string) error {
	s := strings.TrimSpace(v)
	switch strings.ToLower(s) {
	case "", "unset", "clear":
		// States nothing: the model keeps the templates it has.
		c.Notices = nil
		return nil
	case "none", "default", "off", "detach":
		c.Notices = &xollama.Notices{Templates: xollama.NoticeTemplatesNone}
		return nil
	}
	var digest string
	if err := setBlob(s, &digest); err != nil {
		return err
	}
	// The shape is checked here, where the file is; the engine judges the
	// templates themselves when the server attaches them.
	if path, local := mediaFiles[digest]; local {
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if _, err := xollama.ParseNoticeTemplates(data); err != nil {
			return err
		}
	}
	c.Notices = &xollama.Notices{Templates: digest}
	return nil
}

// uploadNoticeTemplates sends the templates file this config attaches, when
// it is a local file the server does not have yet.
func uploadNoticeTemplates(ctx context.Context, client *api.Client, cfg *xollama.Config, out io.Writer) error {
	if cfg.Notices.IsZero() || cfg.Notices.Templates == xollama.NoticeTemplatesNone {
		return nil
	}
	digest := cfg.Notices.Templates
	path, local := mediaFiles[digest]
	if !local {
		return nil
	}
	have, err := client.HeadBlob(ctx, digest)
	if err != nil {
		return fmt.Errorf("notice templates: %w", err)
	}
	if have {
		return nil
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	fmt.Fprintf(out, "   uploading the notice templates (%s)\n", path)
	if err := client.CreateBlob(ctx, digest, f); err != nil {
		return fmt.Errorf("upload the notice templates: %w", err)
	}
	return nil
}

func init() { fields = append(fields, noticeFields()...) }
