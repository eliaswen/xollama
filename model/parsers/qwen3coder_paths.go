package parsers

import "strings"

// A Qwen model writes an array or object parameter as JSON text inside its
// <parameter=...> tag, and copies a Windows path from the prompt as it reads
// it: `[".\run_game.exe manic_miner.html"]`. That is valid JSON whose "\r" is a
// carriage return, so the tool ran "." and then "un_game.exe ..." on a new
// line (pandorum, 2026-10-15, Cerebriline's run_commands). Any of \r \t \n \b
// \f does the same (C:\tmp, .\new, ...); other letters (\U, \m) are invalid
// JSON and already fell back to the raw text (tool-path-backslash hook).
//
// keepPathBackslashes doubles the backslash of such an escape when it sits
// where only a path puts one, so the decoded string keeps the backslash. A
// "\n" in prose or code ("done.\nNext", "a\tb") is left an escape.

// pathEscapeLetters are the JSON escapes that are also the first letter of a
// path segment.
const pathEscapeLetters = "rtnbf"

func keepPathBackslashes(raw string) string {
	if !strings.Contains(raw, `\`) {
		return raw
	}
	var b strings.Builder
	b.Grow(len(raw) + 8)
	inString := false
	start := 0 // where the current string literal's text begins in b
	for i := 0; i < len(raw); i++ {
		c := raw[i]
		if !inString {
			if c == '"' {
				inString = true
				b.WriteByte(c)
				start = b.Len()
				continue
			}
			b.WriteByte(c)
			continue
		}
		switch c {
		case '"':
			inString = false
			b.WriteByte(c)
		case '\\':
			if i+1 >= len(raw) {
				b.WriteByte(c)
				continue
			}
			next := raw[i+1]
			if strings.IndexByte(pathEscapeLetters, next) >= 0 && i+2 < len(raw) && isPathChar(raw[i+2]) && afterPathPrefix(b.String()[start:]) {
				b.WriteString(`\\`)
				b.WriteByte(next)
			} else {
				b.WriteByte(c)
				b.WriteByte(next)
			}
			i++
		default:
			b.WriteByte(c)
		}
	}
	return b.String()
}

// afterPathPrefix reports whether the string text written so far ends where a
// Windows path puts a separator: a relative start (".", ".."), a drive ("C:"),
// or anywhere in a token that already is such a path (".\\", "..\\",
// "\\\\", "C:\\"). Each must begin a token, so "done." or "ratio:" in prose
// does not qualify, and a Unix path, which has no backslash, never does:
// "cd /tmp\nls" keeps its newline.
func afterPathPrefix(text string) bool {
	switch {
	case strings.HasSuffix(text, ".."):
		return tokenStart(text, len(text)-2)
	case strings.HasSuffix(text, "."):
		return tokenStart(text, len(text)-1)
	case len(text) >= 2 && text[len(text)-1] == ':' && isLetter(text[len(text)-2]):
		return tokenStart(text, len(text)-2)
	}
	return windowsPathToken(text)
}

// windowsPathToken reports whether the token that ends text began as a
// Windows path. text is raw JSON, so a separator in it is the two bytes \\.
func windowsPathToken(text string) bool {
	tok := text[strings.LastIndexAny(text, tokenDelimiters)+1:]
	return strings.HasPrefix(tok, `.\\`) || strings.HasPrefix(tok, `..\\`) || strings.HasPrefix(tok, `\\`) ||
		(len(tok) >= 4 && isLetter(tok[0]) && tok[1] == ':' && tok[2:4] == `\\`)
}

// tokenDelimiters end a token in a shell command or an argument list.
const tokenDelimiters = " \t'=(\"&;|"

// tokenStart reports whether position i of text begins a token.
func tokenStart(text string, i int) bool {
	if i == 0 {
		return true
	}
	return strings.IndexByte(tokenDelimiters, text[i-1]) >= 0
}

func isLetter(c byte) bool { return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' }

func isPathChar(c byte) bool {
	return isLetter(c) || c >= '0' && c <= '9' || c == '_' || c == '-' || c == '.'
}
