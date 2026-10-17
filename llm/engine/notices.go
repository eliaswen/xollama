package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
)

// xollama-hook: notices — see docs/xollama/notices.mdx.

// noticeConfigTimeout bounds one --print-notice-config run: it loads no model
// and touches no GPU, so it is the engine's start and nothing else.
const noticeConfigTimeout = 60 * time.Second

// NoticeConfigCommand is the command line that has the engine print its
// effective generation_notice block, after file when one is given, and exit:
// 0 with the block on stdout, 1 with the reason as its last stderr line.
func NoticeConfigCommand(artifact, file, goos string) (string, []string) {
	args := []string{"--server", "--print-notice-config"}
	if file != "" {
		args = append(args, "--notice-templates", file)
	}
	return run(artifact, args, goos)
}

// PrintNoticeConfig runs NoticeConfigCommand on the engine this server would
// start, with its payload HOME, and returns the block. A refused file is the
// engine's own one-line reason ("notice-templates: think[2].template: ...").
func PrintNoticeConfig(ctx context.Context, libOllamaPath, file string) ([]byte, error) {
	home, _ := os.UserHomeDir()
	artifact, err := Find(envconfig.Var(EnvPath), DefaultDirs(libOllamaPath, home))
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, noticeConfigTimeout)
	defer cancel()
	name, args := NoticeConfigCommand(artifact, file, runtime.GOOS)
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Env = envconfig.Environ()
	if payloadHome := PayloadHome(artifact, libOllamaPath, home); payloadHome != "" {
		cmd.Env = append(cmd.Env, "HOME="+payloadHome)
		defer AdoptPayloadHome(payloadHome)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, noticeConfigError(stderr.String())
	}
	return out, nil
}

// noticeConfigError is the engine's last stderr line: the field and the
// reason for a refused file. An engine without contract v2 does not know the
// flag, and its last line says so in its own words.
func noticeConfigError(stderr string) error {
	lines := strings.Split(strings.TrimSpace(stderr), "\n")
	if l := strings.TrimSpace(lines[len(lines)-1]); l != "" {
		return errors.New(l)
	}
	return errors.New("the engine printed no notice configuration")
}

var noticesExplicit = struct {
	sync.Mutex
	path string
	ok   bool
	v3   bool
	bar  bool
	done bool
}{}

// NoticesServed reports whether the engine this server starts plants notices
// under contract v2 and takes --notice-templates. The pinned engine answers
// from its pin; an operator's XOLLAMA_ENGINE_PATH, which the pin does not
// describe, is asked once with --print-notice-config. An engine that does
// not know the flag would refuse a whole command line carrying it.
func NoticesServed(libOllamaPath string) bool {
	return noticesServed(libOllamaPath).ok
}

// NoticesV3Served reports whether that engine also takes contract v3: the
// pin's generation_notice_v3, or a printed block with a "v3" member.
func NoticesV3Served(libOllamaPath string) bool {
	s := noticesServed(libOllamaPath)
	return s.ok && s.v3
}

// NoticesCopyBarServed reports whether that engine takes v3's `copy_bar`: the
// pin's generation_notice_copy_bar, or a printed v3 block with `copy_bar`.
func NoticesCopyBarServed(libOllamaPath string) bool {
	s := noticesServed(libOllamaPath)
	return s.ok && s.v3 && s.bar
}

type noticeSupport struct{ ok, v3, bar bool }

func noticesServed(libOllamaPath string) noticeSupport {
	explicit := envconfig.Var(EnvPath)
	if explicit == "" {
		return noticeSupport{HasGenerationNotice(), HasGenerationNoticeV3(), HasGenerationNoticeCopyBar()}
	}
	noticesExplicit.Lock()
	defer noticesExplicit.Unlock()
	if !noticesExplicit.done || noticesExplicit.path != explicit {
		out, err := PrintNoticeConfig(context.Background(), libOllamaPath, "")
		noticesExplicit.path, noticesExplicit.ok, noticesExplicit.done = explicit, err == nil, true
		noticesExplicit.v3, noticesExplicit.bar = false, false
		if err == nil {
			noticesExplicit.v3, noticesExplicit.bar = printedV3(out)
		}
	}
	return noticeSupport{noticesExplicit.ok, noticesExplicit.v3, noticesExplicit.bar}
}

// printedV3 reports whether a --print-notice-config block carries contract
// v3's forms, at its top level or under generation_notice, and whether they
// include the copy bar's defaults.
func printedV3(out []byte) (v3, copyBar bool) {
	type block struct {
		V3 *struct {
			CopyBar json.RawMessage `json:"copy_bar"`
		} `json:"v3"`
	}
	var b struct {
		block
		GenerationNotice block `json:"generation_notice"`
	}
	if json.Unmarshal(out, &b) != nil {
		return false, false
	}
	for _, v := range []block{b.block, b.GenerationNotice} {
		if v.V3 != nil {
			return true, len(v.V3.CopyBar) > 0
		}
	}
	return false, false
}
