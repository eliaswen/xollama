package manifest

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"

	"github.com/ollama/ollama/types/model"
)

// Since upstream v0.40.0 a named manifest lives in manifests-v2/, and a write
// of a single manifest removes its legacy copy in manifests/. Upstream keeps
// that copy (a downgrade anchor) only after a pull and a manifest-list
// create. A pre-v0.40 ollama on the same store reads manifests/ alone, so a
// model created or rewritten here -- every `tweak model` save -- vanished
// from its list, and its prune at the next start deleted the model's blobs:
// on pandorum, 21 blobs of seven models, the GGUFs among them, 2026-10-15
// (shared-store hook).
//
// keepLegacyManifest writes the legacy copy instead, as the anchor upstream
// writes after a pull: the manifest's own layers, so the older ollama lists and
// loads the model, plus the manifest blob, so its prune keeps that too. A
// manifest list is left to its callers, which write their own anchor, and a
// name the legacy tree cannot hold has no copy.
func keepLegacyManifest(name model.Name, data []byte, digest string) error {
	if err := removeLegacyManifestPaths(name); err != nil {
		return err
	}

	mf, ok := singleManifest(data)
	if !ok {
		return nil
	}

	blob, err := canonicalBlobDigest(digest)
	if err != nil {
		return err
	}
	anchor := mf
	anchor.Layers = append(append([]Layer(nil), mf.Layers...), Layer{
		MediaType: MediaTypeManifest,
		Digest:    blob,
		Size:      int64(len(data)),
	})
	out, err := json.Marshal(anchor)
	if err != nil {
		return err
	}
	return writeLegacyManifestData(name, out)
}

// singleManifest decodes data when it is a single manifest; bytes that are not
// a manifest, or are a manifest list, have no legacy form.
func singleManifest(data []byte) (Manifest, bool) {
	var mf Manifest
	if err := json.Unmarshal(data, &mf); err != nil {
		return Manifest{}, false
	}
	return mf, mf.MediaType != MediaTypeManifestList && len(mf.Manifests) == 0
}

// legacyAnchorCurrent reports whether data is a legacy anchor of the manifest
// n names in manifests-v2/ now: one of its layers is that manifest's blob.
func legacyAnchorCurrent(n model.Name, data []byte) bool {
	var mf Manifest
	if err := json.Unmarshal(data, &mf); err != nil {
		return false
	}
	path, err := V2PathForName(normalizeLogicalName(n))
	if err != nil {
		return false
	}
	current, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(current))
	for _, l := range mf.Layers {
		if (l.MediaType == MediaTypeManifest || l.MediaType == MediaTypeManifestList) && l.Digest == digest {
			return true
		}
	}
	return false
}
