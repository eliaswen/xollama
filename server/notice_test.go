package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func withNoticesServed(t *testing.T, on bool) {
	t.Helper()
	old := noticesServed
	noticesServed = func() bool { return on }
	t.Cleanup(func() { noticesServed = old })
}

func identity(t *testing.T) api.XollamaIdentity {
	t.Helper()
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.GET(api.XollamaIdentityPath, XollamaIdentityHandler)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, api.XollamaIdentityPath, nil))
	var id api.XollamaIdentity
	if err := json.Unmarshal(w.Body.Bytes(), &id); err != nil {
		t.Fatal(err)
	}
	return id
}

// A client takes the spans it hides only from the block: without it the
// planted notices would show in its chat (ollama #1449), so the feature never
// goes out alone.
func TestNoticesAreAdvertisedWithTheirHideSpansOnlyWhenTheEngineHasThem(t *testing.T) {
	withNoticesServed(t, true)
	id := identity(t)
	if !slices.Contains(id.Features, api.FeatureGenerationNotice) {
		t.Fatalf("features %v lack %s", id.Features, api.FeatureGenerationNotice)
	}
	g := id.GenerationNotice
	if g == nil || g.Version != 2 || len(g.Hide) != 1 || g.Hide[0].Open != "<engine_status" || g.Hide[0].Close == nil || *g.Hide[0].Close != "</engine_status>" {
		t.Fatalf("generation_notice = %+v, want version 2 hiding <engine_status ... </engine_status>", g)
	}

	withNoticesServed(t, false)
	id = identity(t)
	if slices.Contains(id.Features, api.FeatureGenerationNotice) || id.GenerationNotice != nil {
		t.Fatalf("an engine without notices advertised them: %v %+v", id.Features, id.GenerationNotice)
	}
}

// The hook line in ChatHandler: the client's object reaches the runner
// unchanged.
func TestAChatRequestsNoticesReachTheRunner(t *testing.T) {
	mock := &mockRunner{}
	mock.CompletionFn = func(ctx context.Context, r llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
		fn(llm.CompletionResponse{Content: "hi"})
		fn(llm.CompletionResponse{Done: true, DoneReason: llm.DoneReasonStop})
		return nil
	}
	gin.SetMode(gin.TestMode)
	s := newServerWithMockRunner(t, mock)
	createMinimalGGUFModel(t, s, "nt", nil, "{{ .Prompt }}", nil)

	notices := json.RawMessage(`{"key":"s:1","answer":[{"at_percent":80}]}`)
	w := createRequest(t, s.ChatHandler, api.ChatRequest{Model: "nt", Messages: []api.Message{{Role: "user", Content: "x"}}, Notices: notices})
	if w.Code != http.StatusOK {
		t.Fatalf("status %d %s", w.Code, w.Body)
	}
	if string(mock.CompletionRequest.Notices) != string(notices) {
		t.Fatalf("runner got notices %q, want %q", mock.CompletionRequest.Notices, notices)
	}
}

func TestTheWaitingCallRefusesWhatContractV1DoesNotHave(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := &Server{}
	r := gin.New()
	r.POST(api.XollamaNoticePath, s.NoticeHandler)
	for body, want := range map[string]int{
		`{"key":"s:1","kind":"message_text"}`:     http.StatusBadRequest,
		`{"kind":"message_waiting"}`:              http.StatusBadRequest,
		`not json`:                                http.StatusBadRequest,
		`{"key":"gone","kind":"message_waiting"}`: http.StatusNotFound,
	} {
		w := httptest.NewRecorder()
		r.ServeHTTP(w, httptest.NewRequest(http.MethodPost, api.XollamaNoticePath, strings.NewReader(body)))
		if w.Code != want {
			t.Errorf("%s: status %d, want %d (%s)", body, w.Code, want, w.Body)
		}
	}
}

// context_window_unbooked_v1 is advertised exactly while num_ctx_hold reaches
// the engine.
func TestUnbookedWindowsAreAdvertisedOnlyWhenTheEngineTakesThem(t *testing.T) {
	old := unbookedServed
	t.Cleanup(func() { unbookedServed = old })

	unbookedServed = func() bool { return true }
	if f := identity(t).Features; !slices.Contains(f, api.FeatureContextWindowUnbooked) {
		t.Fatalf("features %v lack %s", f, api.FeatureContextWindowUnbooked)
	}
	unbookedServed = func() bool { return false }
	if f := identity(t).Features; slices.Contains(f, api.FeatureContextWindowUnbooked) {
		t.Fatalf("an engine without num_ctx_hold advertised %s", api.FeatureContextWindowUnbooked)
	}
}
