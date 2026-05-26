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

package pollingsource

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/interface/plugin"
)

// fakeCollector counts how many times Poll has been called.
type fakeCollector struct {
	name      string
	pollCount atomic.Int64
}

func (f *fakeCollector) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "fake-collector", Name: f.name}
}

func (f *fakeCollector) Poll(_ context.Context) (any, error) {
	f.pollCount.Add(1)
	return nil, nil
}

func TestNew_EmptyName(t *testing.T) {
	_, err := New("")
	if err == nil {
		t.Error("expected error for empty name, got nil")
	}
}

func TestStart_AlreadyStarted(t *testing.T) {
	src, err := New("test")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx := context.Background()
	if err := src.Start(ctx); err != nil {
		t.Fatalf("first Start: %v", err)
	}
	defer src.Stop()

	if err := src.Start(ctx); err == nil {
		t.Error("expected error on second Start, got nil")
	}
}

func TestCollectorPolled(t *testing.T) {
	src, _ := New("test")
	c := &fakeCollector{name: "c1"}
	src.RegisterCollector(c, 10*time.Millisecond)

	if err := src.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	time.Sleep(50 * time.Millisecond)
	src.Stop()

	if got := c.pollCount.Load(); got < 2 {
		t.Errorf("expected >= 2 polls, got %d", got)
	}
}

func TestMultipleCollectors_DifferentFrequencies(t *testing.T) {
	src, _ := New("test")
	fast := &fakeCollector{name: "fast"}
	slow := &fakeCollector{name: "slow"}
	src.RegisterCollector(fast, 10*time.Millisecond)
	src.RegisterCollector(slow, 30*time.Millisecond)

	if err := src.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	time.Sleep(60 * time.Millisecond)
	src.Stop()

	fastCount := fast.pollCount.Load()
	slowCount := slow.pollCount.Load()
	if fastCount <= slowCount {
		t.Errorf("expected fast collector (%d) to be polled more than slow (%d)", fastCount, slowCount)
	}
}
