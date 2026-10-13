package llm

// xollama: an opencoti completion that fails mid-stream is an error, not an
// empty answer -- the `engine-stream-error` hook in Completion.
//
// Upstream's completion loop reads `content` and `stop` from each event and
// nothing else, so the engine's own error event
// (`data: {"error":{"code":500,"message":"vector","type":"server_error"}}`)
// parses as an event with no content, the stream ends with no final event,
// and Completion returns nil without having called back once. The client then
// gets 200 with a zero response (`"done":false`, no model, no message), or an
// empty stream. Measured on the Mac mini with v0.40.3-xollama.1 and a
// Qwen3.5-family model at temperature 0 (opencoti bug-4048: the engine logs
// `got exception: vector`, `srv send_error`). The chat loop already returns
// such an event as an error; stock llama.cpp keeps upstream's behaviour here.

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// streamErrorMessage is the message of an engine error event, or "".
func streamErrorMessage(evt []byte) string {
	var e struct {
		Error json.RawMessage `json:"error"`
	}
	if json.Unmarshal(evt, &e) != nil || len(e.Error) == 0 || string(e.Error) == "null" {
		return ""
	}
	var obj struct {
		Message string `json:"message"`
	}
	if json.Unmarshal(e.Error, &obj) == nil && obj.Message != "" {
		return obj.Message
	}
	var text string
	if json.Unmarshal(e.Error, &text) == nil && text != "" {
		return text
	}
	return strings.TrimSpace(string(e.Error))
}

// streamFailed is the completion loop's: on opencoti, an error event ends the
// completion with the engine's message.
func (s *llamaServerRunner) streamFailed(evt []byte) error {
	if !s.usedOpencoti {
		return nil
	}
	msg := streamErrorMessage(evt)
	if msg == "" {
		return nil
	}
	return fmt.Errorf("an error was encountered while running the model: %s", msg)
}

// streamUnfinished is the completion loop's last word: on opencoti, a stream
// that ended cleanly without its final event did not finish, whatever it
// delivered before.
func (s *llamaServerRunner) streamUnfinished() error {
	if !s.usedOpencoti {
		return nil
	}
	return errors.New("an error was encountered while running the model: the engine ended the response before finishing it")
}
