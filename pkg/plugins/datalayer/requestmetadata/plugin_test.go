/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package requestmetadata

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/datalayer"
)

// fakeDataStore is an in-memory DataStore for tests.
type fakeDataStore struct {
	mu     sync.Mutex
	models map[string]datalayer.Model
}

func newFakeDataStore() *fakeDataStore {
	return &fakeDataStore{models: make(map[string]datalayer.Model)}
}

func (f *fakeDataStore) GetOrCreateModel(name string) datalayer.Model {
	f.mu.Lock()
	defer f.mu.Unlock()
	if m, ok := f.models[name]; ok {
		return m
	}
	m := datalayer.NewModel(name)
	f.models[name] = m
	return m
}

// makeRequestEvent creates a RequestEventType event with model and max_tokens.
func makeRequestEvent(model string, maxTokens float64) datalayer.Event {
	req := framework.NewInferenceRequest()
	req.Body["model"] = model
	req.Body["max_tokens"] = maxTokens
	return datalayer.Event{
		Type:    datalayer.RequestEventType,
		Payload: datalayer.RequestPayload{Request: req},
	}
}

// makeResponseEvent creates a ResponseEventType event with model, duration, and max_tokens.
// maxTokens mirrors the original request's max_tokens so the extractor can decrement correctly.
func makeResponseEvent(model string, durationMs int, maxTokens float64) datalayer.Event {
	req := framework.NewInferenceRequest()
	req.Body["model"] = model
	req.Body["max_tokens"] = maxTokens
	return datalayer.Event{
		Type: datalayer.ResponseEventType,
		Payload: datalayer.ResponsePayload{
			Request:  req,
			Response: framework.NewInferenceResponse(),
			Duration: time.Duration(durationMs) * time.Millisecond,
		},
	}
}

// getRequestMetadata asserts the request-metadata attribute exists for model and returns it.
func getRequestMetadata(t testing.TB, ds *fakeDataStore, model string) RequestMetadataCount {
	t.Helper()
	val, ok := ds.GetOrCreateModel(model).GetAttributes().Get(RequestMetadataAttributeKey)
	if !ok {
		t.Fatalf("expected %q attribute for model %q", RequestMetadataAttributeKey, model)
	}
	rc, ok := val.(RequestMetadataCount)
	if !ok {
		t.Fatalf("expected RequestMetadataCount for model %q", model)
	}
	return rc
}

func newRequestMetadataTest(t *testing.T) (*RequestMetadataExtractor, *fakeDataStore) {
	t.Helper()
	ds := newFakeDataStore()
	return NewRequestMetadataExtractor(ds), ds
}

func TestRequestIncrementsCounter(t *testing.T) {
	ext, ds := newRequestMetadataTest(t)

	batch := []datalayer.Event{makeRequestEvent("m1", 100)}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	rc := getRequestMetadata(t, ds, "m1")
	if rc.Requests != 1 {
		t.Errorf("expected Requests=1, got %d", rc.Requests)
	}
	if rc.Tokens != 100 {
		t.Errorf("expected Tokens=100, got %d", rc.Tokens)
	}
}

func TestResponseDecrementsCounter(t *testing.T) {
	ext, ds := newRequestMetadataTest(t)

	// Response carries the original request's max_tokens so the extractor can decrement correctly.
	batch := []datalayer.Event{
		makeRequestEvent("m1", 100),
		makeResponseEvent("m1", 50, 100),
	}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	rc := getRequestMetadata(t, ds, "m1")
	if rc.Requests != 0 {
		t.Errorf("expected Requests=0, got %d", rc.Requests)
	}
	if rc.Tokens != 0 {
		t.Errorf("expected Tokens=0, got %d", rc.Tokens)
	}
}

func TestCounterFloorsAtZero(t *testing.T) {
	ext, ds := newRequestMetadataTest(t)

	// Response with no prior request — both counters must floor at zero.
	batch := []datalayer.Event{makeResponseEvent("m1", 50, 100)}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	rc := getRequestMetadata(t, ds, "m1")
	if rc.Requests != 0 {
		t.Errorf("expected Requests=0, got %d", rc.Requests)
	}
	if rc.Tokens != 0 {
		t.Errorf("expected Tokens=0, got %d", rc.Tokens)
	}
}

func TestRequestMetadataMultipleModels(t *testing.T) {
	ext, ds := newRequestMetadataTest(t)

	batch := []datalayer.Event{
		makeRequestEvent("m1", 10),
		makeRequestEvent("m2", 20),
	}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	rc1 := getRequestMetadata(t, ds, "m1")
	if rc1.Requests != 1 || rc1.Tokens != 10 {
		t.Errorf("m1: expected {Requests:1, Tokens:10}, got %+v", rc1)
	}

	rc2 := getRequestMetadata(t, ds, "m2")
	if rc2.Requests != 1 || rc2.Tokens != 20 {
		t.Errorf("m2: expected {Requests:1, Tokens:20}, got %+v", rc2)
	}
}

func TestRequestMetadataUnknownEventTypeIgnored(t *testing.T) {
	ext, ds := newRequestMetadataTest(t)

	batch := []datalayer.Event{{Type: "unknown"}}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	ds.mu.Lock()
	modelCount := len(ds.models)
	ds.mu.Unlock()
	if modelCount != 0 {
		t.Errorf("expected no models in datastore, got %d", modelCount)
	}
}

func TestRequestMetadataMissingModelFieldIgnored(t *testing.T) {
	ext, ds := newRequestMetadataTest(t)

	// Payload without a "model" key — no counter should be updated.
	req := framework.NewInferenceRequest()
	req.Body["max_tokens"] = float64(50)
	batch := []datalayer.Event{
		{Type: datalayer.RequestEventType, Payload: datalayer.RequestPayload{Request: req}},
	}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	ds.mu.Lock()
	modelCount := len(ds.models)
	ds.mu.Unlock()
	if modelCount != 0 {
		t.Errorf("expected no models in datastore, got %d", modelCount)
	}
}
