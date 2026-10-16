package server

import (
	"fmt"
	"io"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/parser"
)

// A Modelfile whose TEMPLATE quote was never closed has been refused by the
// parser since 2026-09-29, but a model created before that, or by another
// ollama on the same store, keeps the swallowed RENDERER, PARSER and
// PARAMETER lines in its template layer. Every later create FROM it, a
// `tweak model` save included, carried that layer on without a word: on
// pandorum, omnimerge-v4-mtp_tb:27b-iq2m-128k ran for weeks with no renderer
// and no parser while `show` printed both inside its template
// (template-swallowed hook).

// carriedLayerKinds are the inherited layers whose text the check reads, by
// the parser's name for them.
var carriedLayerKinds = map[string]string{
	manifest.MediaTypeImageTemplate: "template",
	manifest.MediaTypeImageSystem:   "system",
}

// refuseSwallowedCarry refuses a create that carries on a TEMPLATE or SYSTEM
// layer holding Modelfile directives. A request that sets its own value
// replaces the layer, so it is never refused for the old one.
func refuseSwallowedCarry(layers []manifest.Layer, template, system string, from string) error {
	for _, l := range layers {
		kind, ok := carriedLayerKinds[l.MediaType]
		if !ok || (kind == "template" && template != "") || (kind == "system" && system != "") {
			continue
		}
		text, err := layerText(l)
		if err != nil {
			return err
		}
		if line := parser.SwallowedDirective(kind, text); line != "" {
			return fmt.Errorf("the %s this model inherits from %s contains the Modelfile line %q: its quote was never closed when the model was first created, so that line and the ones after it are %s text, not settings; recreate it with the %s quote closed (xollama show %s --modelfile, fix it, xollama create)", strings.ToUpper(kind), from, line, kind, strings.ToUpper(kind), from)
		}
	}
	return nil
}

// warnSwallowedTemplate logs a model whose template already holds directives,
// once per load, so a model created before the check is named in the log.
func warnSwallowedTemplate(m *Model) {
	if m == nil || m.Template == nil {
		return
	}
	if line := parser.SwallowedDirective("template", m.Template.String()); line != "" {
		slog.Warn("the model's TEMPLATE contains a Modelfile directive: its quote was never closed at create, so the directive is template text and not applied", "model", m.ShortName, "line", line)
	}
}

func layerText(l manifest.Layer) (string, error) {
	r, err := l.Open()
	if err != nil {
		return "", err
	}
	defer r.Close()
	data, err := io.ReadAll(io.LimitReader(r, 1<<20))
	if err != nil {
		return "", err
	}
	return string(data), nil
}
