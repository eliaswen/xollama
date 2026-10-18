package systel

func platformReaders(root int) []reader {
	return []reader{
		&procReader{root: root, p: linuxPlatform{procfs: "/proc"}},
		nvmlReader{},
		&drmReader{sysfs: "/sys"},
	}
}
