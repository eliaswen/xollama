package server

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/llm/engine"
)

// xollama-hook: notices — see docs/xollama/notices.mdx.

// noticesServed reports whether this server offers generation notices: the
// engine it ships declares them and is not switched off for stock llama.cpp.
// A client reads it from /api/xollama and sends `notices` only when it is.
var noticesServed = func() bool {
	return engine.HasGenerationNotice() && !strings.EqualFold(envconfig.Engine(), "llamacpp")
}

// generationNotice is the capability block /api/xollama carries beside the
// feature: the client takes the marker it hides from here, and without the
// block the planted lines would show in its chat.
func generationNotice() *api.GenerationNotice {
	if !noticesServed() {
		return nil
	}
	return &api.GenerationNotice{Version: 1, Marker: engine.NoticeMarker}
}

// NoticeHandler serves POST /api/xollama/notice: a client says a message is
// waiting for the running request that carries this key, and the engine
// plants a notice in it. No message text reaches the model. The answer is the
// engine's own: 200, 400 for an unknown kind, 404 when no running request has
// the key (the client ignores it), 409 for a key two requests share.
func (s *Server) NoticeHandler(c *gin.Context) {
	var req api.NoticeRequest
	if err := json.NewDecoder(c.Request.Body).Decode(&req); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid request body: " + err.Error()})
		return
	}
	if req.Key == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "key is required"})
		return
	}
	if req.Kind != api.NoticeKindMessageWaiting {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "unknown kind " + `"` + req.Kind + `"` + `; contract v1 has "message_waiting"`})
		return
	}
	status, body, err := llm.PostNotice(c.Request.Context(), req.Key, req.Kind)
	switch {
	case errors.Is(err, llm.ErrNoNoticeKey):
		c.AbortWithStatusJSON(http.StatusNotFound, gin.H{"error": err.Error()})
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadGateway, gin.H{"error": "engine: " + err.Error()})
	case json.Valid(body):
		c.Data(status, "application/json", body)
	default:
		c.JSON(status, gin.H{"error": strings.TrimSpace(string(body))})
	}
}
