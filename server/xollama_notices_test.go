package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/xollama"
)

const customNotices = `{"answer":[{"at_percent":75,"template":"[note {left} of {total} left] finish [/note]"}]}`

func withNoticeCheck(t *testing.T, check func(context.Context, string) error) {
	t.Helper()
	old := noticeConfigCheck
	noticeConfigCheck = check
	t.Cleanup(func() { noticeConfigCheck = old })
}

func noticeBlob(t *testing.T, data string) string {
	t.Helper()
	l, err := manifest.NewLayer(bytes.NewReader([]byte(data)), xollama.MediaTypeNotices)
	if err != nil {
		t.Fatal(err)
	}
	return l.Digest
}

// What `xollama tweak model --notice-templates FILE` does: upload the file,
// then create the model FROM itself naming it. The engine is asked first; the
// layer is the truth afterwards, through the next save, the launch and show.
func TestAModelsNoticeTemplatesAreALayerTheEngineStartsWith(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	withNoticesServed(t, true)
	var checked string
	withNoticeCheck(t, func(_ context.Context, file string) error { checked = file; return nil })
	var s Server

	_, digest := createBinFile(t, nil, nil)
	if w := createRequest(t, s.CreateHandler, api.CreateRequest{Name: "nm", Files: map[string]string{"m.gguf": digest}, Stream: &stream}); w.Code != http.StatusOK {
		t.Fatalf("seed: %d %s", w.Code, w.Body)
	}

	m, err := GetModel("nm")
	if err != nil {
		t.Fatal(err)
	}
	def := noticeTemplatesPath(m)
	if data, err := os.ReadFile(def); err != nil || !bytes.Equal(data, xollama.DefaultNoticeTemplates) {
		t.Fatalf("a model without templates does not start with the defaults: %q %v", def, err)
	}

	tmpl := noticeBlob(t, customNotices)
	w := createRequest(t, s.CreateHandler, api.CreateRequest{Name: "nm", From: "nm", Xollama: &xollama.Config{Notices: &xollama.Notices{Templates: tmpl}}, Stream: &stream})
	if w.Code != http.StatusOK || strings.Contains(w.Body.String(), "error") {
		t.Fatalf("attach: %d %s", w.Code, w.Body)
	}
	// The parts the file left out are xollama's defaults, so the spans cover
	// the thinking ladder and the waiting message too.
	m, err = GetModel("nm")
	if err != nil {
		t.Fatal(err)
	}
	blob := m.NoticeTemplatesPath
	if blob == "" || checked != blob {
		t.Fatalf("the engine checked %q, the model carries %q", checked, blob)
	}

	// A later save that says nothing about notices keeps them.
	if w := createRequest(t, s.CreateHandler, api.CreateRequest{Name: "nm", From: "nm", System: "be brief", Stream: &stream}); w.Code != http.StatusOK {
		t.Fatalf("resave: %d %s", w.Code, w.Body)
	}
	if m, err = GetModel("nm"); err != nil {
		t.Fatal(err)
	}
	if got := llamaServerConfigForModel(m).NoticeTemplatesPath; got != blob {
		t.Fatalf("launch --notice-templates %q, want the model's layer %q", got, blob)
	}
	if m.Xollama != nil && m.Xollama.Notices != nil {
		t.Fatalf("the notices block was stored in the config: %+v", m.Xollama.Notices)
	}
	g := modelGenerationNotice(m)
	if g == nil || len(g.Hide) != 2 || g.Hide[0].Open != "<engine_status" || g.Hide[1].Open != "[note" || g.Hide[1].Close == nil || *g.Hide[1].Close != "[/note]" {
		t.Fatalf("show's generation_notice = %+v, want the default thinking ladder's span and [note ... [/note]", g)
	}
	if !strings.Contains(string(g.Templates), "[note {left} of {total} left]") || !strings.Contains(string(g.Templates), "A new message is waiting") {
		t.Fatalf("the stored templates are not the file completed with the defaults: %s", g.Templates)
	}

	// none goes back to the defaults.
	if w := createRequest(t, s.CreateHandler, api.CreateRequest{Name: "nm", From: "nm", Xollama: &xollama.Config{Notices: &xollama.Notices{Templates: xollama.NoticeTemplatesNone}}, Stream: &stream}); w.Code != http.StatusOK {
		t.Fatalf("detach: %d %s", w.Code, w.Body)
	}
	if m, _ = GetModel("nm"); m.NoticeTemplatesPath != "" {
		t.Fatalf("none left the layer: %q", m.NoticeTemplatesPath)
	}
}

// A file the engine refuses is refused at the save, with the engine's own
// line, never attached to fail every load later.
func TestTemplatesTheEngineRefusesAreNotAttached(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	withNoticesServed(t, true)
	reason := "notice-templates: answer[0].template: a template that spans lines must contain its closing tag (e.g. '</budget>')"
	withNoticeCheck(t, func(context.Context, string) error { return errors.New(reason) })
	var s Server

	_, digest := createBinFile(t, nil, nil)
	if w := createRequest(t, s.CreateHandler, api.CreateRequest{Name: "nr", Files: map[string]string{"m.gguf": digest}, Stream: &stream}); w.Code != http.StatusOK {
		t.Fatalf("seed: %d %s", w.Code, w.Body)
	}
	tmpl := noticeBlob(t, `{"answer":[{"at_percent":75,"template":"<budget>\nline two"}]}`)
	w := createRequest(t, s.CreateHandler, api.CreateRequest{Name: "nr", From: "nr", Xollama: &xollama.Config{Notices: &xollama.Notices{Templates: tmpl}}, Stream: &stream})
	var body struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil || !strings.Contains(body.Error, reason) {
		t.Fatalf("the refusal does not carry the engine's line: %d %s", w.Code, w.Body)
	}
	if m, _ := GetModel("nr"); m.NoticeTemplatesPath != "" {
		t.Fatalf("a refused file was attached: %q", m.NoticeTemplatesPath)
	}
}
