package server

import (
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/internal/systel"
	"github.com/ollama/ollama/ml"
)

// systemInterval and systemIdle: the snapshot is refreshed every interval
// while it was polled within idle.
const (
	systemInterval = 2 * time.Second
	systemIdle     = time.Minute
)

var (
	systemOnce    sync.Once
	systemSampler *systel.Sampler
	// systemSnapshot and knownDevices are the readings and the server's own
	// device list, behind seams for tests.
	systemSnapshot = func() systel.Snapshot {
		systemOnce.Do(func() { systemSampler = systel.New(systemInterval, systemIdle) })
		return systemSampler.Get()
	}
	knownDevices = discover.KnownDevices
)

// SystemHandler answers /api/xollama/system from the cached snapshot. It
// never starts a discovery, an engine or a probe: the vendor readings come
// from systel, the server's view of each GPU from the last discovery, and the
// models from the scheduler without waiting on a load.
func (s *Server) SystemHandler(c *gin.Context) {
	c.JSON(http.StatusOK, s.systemResponse(systemSnapshot(), knownDevices()))
}

func (s *Server) systemResponse(snap systel.Snapshot, known []ml.DeviceInfo) api.SystemResponse {
	resp := api.SystemResponse{
		AsOf:       snap.At,
		IntervalMS: int(systemInterval / time.Millisecond),
		Host:       snap.Host,
		Server:     snap.Server,
		Devices:    make([]api.SystemDevice, 0, len(snap.Devices)),
		Models:     []api.SystemModel{},
	}
	resp.Server.Children = append([]api.SystemProcess(nil), snap.Server.Children...)

	view := xollamaDevices(known).Devices
	for _, d := range snap.Devices {
		d.Link = withMeasuredLink(d.PCIID, d.Link)
		if k, ok := knownFor(view, d); ok {
			d.Backend, d.ID, d.Engine = k.Backend, k.ID, k.Engine
			d.Integrated = d.Integrated || k.Integrated
			if d.Name == "" {
				d.Name = k.Description
			}
			// Unified memory has no size of its own: the working set the
			// server already plans against is the GPU's share.
			if d.TotalMemory == nil && k.TotalMemory > 0 {
				d.TotalMemory = ptrTo(k.TotalMemory)
				if d.UsedMemory != nil && *d.UsedMemory <= k.TotalMemory {
					d.FreeMemory = ptrTo(k.TotalMemory - *d.UsedMemory)
				}
			}
		}
		resp.Devices = append(resp.Devices, d)
	}

	if s.sched != nil {
		models := s.sched.systemModels(view)
		byPID := map[int]string{}
		for _, m := range models {
			if m.PID > 0 {
				byPID[m.PID] = m.Name
			}
		}
		for i := range resp.Server.Children {
			resp.Server.Children[i].Model = byPID[resp.Server.Children[i].PID]
		}
		resp.Models = models
	}
	return resp
}

// knownFor finds the server's own entry for a GPU: by bus address, and on a
// machine with one GPU of a vendor that has none (Apple), by being the only
// one.
func knownFor(view []api.XollamaDevice, d api.SystemDevice) (api.XollamaDevice, bool) {
	if d.PCIID != "" {
		for _, k := range view {
			if strings.EqualFold(k.PCIID, d.PCIID) {
				return k, true
			}
		}
		return api.XollamaDevice{}, false
	}
	if d.Vendor == "apple" && len(view) == 1 {
		return view[0], true
	}
	return api.XollamaDevice{}, false
}

func ptrTo[T any](v T) *T { return &v }

// The last link probe's measurements, by bus address: a poll reports them and
// never measures.
var (
	linkMu       sync.Mutex
	linkMeasured = map[string]measuredLink{}
)

type measuredLink struct {
	h2d, d2h float64
	at       time.Time
}

// rememberLinkProbe keeps what a link probe measured.
func rememberLinkProbe(resp api.LinkProbeResponse, at time.Time) {
	linkMu.Lock()
	defer linkMu.Unlock()
	for _, r := range resp.Results {
		if r.Probe == nil {
			continue
		}
		for _, d := range r.Probe.Devices {
			if d.Measured.Available && d.PCIBusID != "" {
				linkMeasured[strings.ToLower(normalizeBusID(d.PCIBusID))] = measuredLink{d.Measured.H2DGBps, d.Measured.D2HGBps, at}
			}
		}
	}
}

func withMeasuredLink(pci string, l *api.SystemLink) *api.SystemLink {
	linkMu.Lock()
	m, ok := linkMeasured[strings.ToLower(pci)]
	linkMu.Unlock()
	if !ok {
		return l
	}
	if l == nil {
		l = &api.SystemLink{}
	} else {
		c := *l
		l = &c
	}
	l.H2DGBps, l.D2HGBps, l.MeasuredAt = ptrTo(m.h2d), ptrTo(m.d2h), ptrTo(m.at)
	return l
}

// normalizeBusID writes the engine's bus id ("00000000:01:00.0") as the
// devices route does ("0000:01:00.0").
func normalizeBusID(s string) string {
	dom, rest, ok := strings.Cut(s, ":")
	if !ok || strings.Count(rest, ":") != 1 {
		return s
	}
	if len(dom) > 4 {
		dom = dom[len(dom)-4:]
	}
	return dom + ":" + rest
}
