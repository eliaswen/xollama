package server

import (
	"net/http"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

// pandorum's omnimerge tag: a template made before create refused an open
// quote, holding the RENDERER, PARSER and PARAMETER lines meant for the model.
const swallowedTemplate = "{{ .Prompt }}\nRENDERER qwen3.5\nPARSER qwen3.5\nPARAMETER repeat_penalty 1\nPARAMETER think_budget_message "

func TestACreateRefusesToCarryOnATemplateThatSwallowedDirectives(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	var s Server

	_, digest := createBinFile(t, nil, nil)
	// The API takes the template as given, as a create before the parser check did.
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:     "broken",
		Files:    map[string]string{"test.gguf": digest},
		Template: swallowedTemplate,
		Stream:   &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("seeding the broken model: status %d: %s", w.Code, w.Body.String())
	}

	// What `tweak model` sends: the model itself as FROM, no template.
	w = createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:   "broken",
		From:   "broken",
		System: "be brief",
		Stream: &stream,
	})
	if w.Code == http.StatusOK && !strings.Contains(w.Body.String(), "RENDERER qwen3.5") {
		t.Fatalf("a create FROM the broken model carried its template on: status %d: %s", w.Code, w.Body.String())
	}
	for _, want := range []string{"RENDERER qwen3.5", "TEMPLATE", "quote was never closed", "xollama show broken --modelfile"} {
		if !strings.Contains(w.Body.String(), want) {
			t.Errorf("the refusal does not say %s: %s", want, w.Body.String())
		}
	}

	// Giving the model its own template is the fix, and is not refused.
	w = createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:     "broken",
		From:     "broken",
		Template: "{{ .Prompt }}",
		Stream:   &stream,
	})
	if w.Code != http.StatusOK || strings.Contains(w.Body.String(), "error") {
		t.Fatalf("a create that replaces the template was refused: status %d: %s", w.Code, w.Body.String())
	}
}
