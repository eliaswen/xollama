//go:build !(linux || windows || darwin) || (darwin && !cgo)

package systel

// Elsewhere only the portable part of the host is read.
type otherPlatform struct{}

func (otherPlatform) host() hostInfo    { return hostInfo{} }
func (otherPlatform) procs() []procInfo { return nil }

func platformReaders(root int) []reader {
	return []reader{&procReader{root: root, p: otherPlatform{}}}
}
