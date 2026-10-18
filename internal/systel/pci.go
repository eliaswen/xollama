package systel

import (
	"fmt"
	"strings"
)

// normalizePCI writes a PCI address as /api/xollama/devices does,
// "0000:01:00.0": NVML pads the domain to eight digits, and Windows numbers
// it all in decimal.
func normalizePCI(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	var dom, bus, dev, fn uint
	if n, _ := fmt.Sscanf(s, "%x:%x:%x.%x", &dom, &bus, &dev, &fn); n == 4 {
		return fmt.Sprintf("%04x:%02x:%02x.%x", dom, bus, dev, fn)
	}
	if n, _ := fmt.Sscanf(s, "%x:%x.%x", &bus, &dev, &fn); n == 3 {
		return fmt.Sprintf("0000:%02x:%02x.%x", bus, dev, fn)
	}
	return s
}
