//go:build cgo

package systel

func platformReaders(root int) []reader {
	return []reader{
		&procReader{root: root, p: darwinPlatform{}},
		iokitReader{},
	}
}
