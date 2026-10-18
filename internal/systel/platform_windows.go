package systel

func platformReaders(root int) []reader {
	return []reader{
		&procReader{root: root, p: windowsPlatform{}},
		nvmlReader{},
		&d3dkmtReader{},
	}
}
