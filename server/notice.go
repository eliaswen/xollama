package server

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/llm/engine"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/xollama"
)

// xollama-hook: notices — see docs/xollama/notices.mdx.

// noticesServed reports whether this server offers generation notices: the
// engine it starts plants them under contract v2 (the pin says so, or an
// operator's XOLLAMA_ENGINE_PATH answers --print-notice-config) and is not
// switched off for stock llama.cpp. A client reads it from /api/xollama and
// sends `notices` only when it is.
var noticesServed = func() bool {
	return !strings.EqualFold(envconfig.Engine(), "llamacpp") && engine.NoticesServed(ml.LibOllamaPath)
}

// noticesV3Served reports whether that engine also takes contract v3.
var noticesV3Served = func() bool {
	return noticesServed() && engine.NoticesV3Served(ml.LibOllamaPath)
}

// generationNotice is the block /api/xollama carries beside the feature: the
// server's default templates and the spans a client hides. A model's own are
// on /api/show (modelGenerationNotice).
func generationNotice() *api.GenerationNotice {
	if !noticesServed() {
		return nil
	}
	return noticeBlock(xollama.DefaultNoticeTemplates)
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
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "unknown kind " + `"` + req.Kind + `"` + `; the one kind is "message_waiting"`})
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

// noticeChunk is the chunk that hands a client one line the engine planted
// under contract v3. It is sent apart, before the chunk the engine reported
// it on: a parser may hold that chunk's text back or drop the chunk, never
// send its text ahead of it, so the line is the next occurrence of
// Text+Break after it in the message, as the contract says.
func noticeChunk(model string, n *api.Notice) (api.ChatResponse, bool) {
	if n == nil {
		return api.ChatResponse{}, false
	}
	return api.ChatResponse{
		Model:     model,
		CreatedAt: time.Now().UTC(),
		Message:   api.Message{Role: "assistant"},
		Notice:    n,
	}, true
}
