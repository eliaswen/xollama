package discover

import "github.com/ollama/ollama/ml"

// KnownDevices is the device list the last discovery left, without running
// one: a reader that must not wait, such as /api/xollama/system, gets nil
// while a discovery holds the list.
func KnownDevices() []ml.DeviceInfo {
	if !deviceMu.TryLock() {
		return nil
	}
	defer deviceMu.Unlock()
	return append([]ml.DeviceInfo(nil), devices...)
}
