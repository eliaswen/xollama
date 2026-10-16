package manifest

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"testing"

	"github.com/ollama/ollama/types/model"
)

// The digests a pre-v0.40 ollama's prune keeps for one model: those its
// legacy manifest lists.
func legacyKept(t *testing.T, n model.Name) []string {
	t.Helper()
	path, err := LegacyPathForName(normalizeLogicalName(n))
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("no legacy manifest for %s, so an older ollama on this store neither lists it nor keeps its blobs: %v", n.DisplayShortest(), err)
	}
	var mf Manifest
	if err := json.Unmarshal(data, &mf); err != nil {
		t.Fatal(err)
	}
	kept := []string{mf.Config.Digest}
	for _, l := range mf.Layers {
		kept = append(kept, l.Digest)
	}
	return kept
}

func TestAManifestWriteKeepsTheCopyAnOlderOllamaReads(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	n := model.ParseName("omnimerge-v4-mtp_tb:27b-iq2m-128k")

	mf := Manifest{
		SchemaVersion: 2,
		MediaType:     MediaTypeManifest,
		Config:        Layer{MediaType: "application/vnd.docker.container.image.v1+json", Digest: "sha256:" + fmt.Sprintf("%064x", 1), Size: 117},
		Layers: []Layer{
			{MediaType: "application/vnd.ollama.image.model", Digest: "sha256:" + fmt.Sprintf("%064x", 2), Size: 10004592448},
			{MediaType: MediaTypeImageTemplate, Digest: "sha256:" + fmt.Sprintf("%064x", 3), Size: 13},
		},
	}
	data, err := json.Marshal(mf)
	if err != nil {
		t.Fatal(err)
	}
	// What a `tweak model` save does: rewrite the model in place.
	for range 2 {
		if err := WriteManifestData(n, data); err != nil {
			t.Fatal(err)
		}
	}

	kept := legacyKept(t, n)
	want := []string{mf.Config.Digest, mf.Layers[0].Digest, mf.Layers[1].Digest, fmt.Sprintf("sha256:%x", sha256.Sum256(data))}
	for _, d := range want {
		if !slices.Contains(kept, d) {
			t.Errorf("an older ollama's prune would delete %s: its legacy manifest lists only %v", d, kept)
		}
	}

	// The copy is not a second model to migrate.
	if migrated, err := MigrateManifestLinks(); err != nil || migrated != 0 {
		t.Fatalf("migration re-imported the kept copy: migrated %d, %v", migrated, err)
	}
	current, err := V2PathForName(n)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := os.ReadFile(current); err != nil || string(got) != string(data) {
		t.Fatalf("the v2 manifest changed: %v %s", err, got)
	}
}

func TestAManifestListWriteLeavesTheAnchorToItsCaller(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	n := model.ParseName("listed")

	list := Manifest{SchemaVersion: 2, MediaType: MediaTypeManifestList, Manifests: []Manifest{{MediaType: MediaTypeManifest}}}
	data, err := json.Marshal(list)
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteManifestData(n, data); err != nil {
		t.Fatal(err)
	}
	path, err := LegacyPathForName(normalizeLogicalName(n))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("a manifest list was copied as a legacy manifest, which an older ollama cannot read: %v", err)
	}
}
