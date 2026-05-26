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
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	dlsrc "github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/interface/datalayer/datasource"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/interface/plugin"
)

const PluginType = "polling-source"

// compile-time interface assertion
var _ dlsrc.PollingSource = &PollingSource{}

type collectorEntry struct {
	collector dlsrc.Collector
	frequency time.Duration
}

type PollingSource struct {
	name       plugin.TypedName
	collectors []collectorEntry
	mu         sync.Mutex

	started atomic.Bool
	cancel  context.CancelFunc
	wg      sync.WaitGroup
}

// Factory is the factory function for PollingSource.
// TODO: configuration to list collectors to enable.
func Factory(name string, _ json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	src, err := New(name)
	if err != nil {
		return nil, fmt.Errorf("failed to create '%s' plugin - %w", PluginType, err)
	}
	return src, nil
}

// New creates a PollingSource that drives each registered Collector at its configured frequency.
func New(name string) (dlsrc.PollingSource, error) {
	if name == "" {
		return nil, fmt.Errorf("name is required for plugin '%s'", PluginType)
	}
	return &PollingSource{
		name: plugin.TypedName{Type: PluginType, Name: name},
	}, nil
}

func (p *PollingSource) TypedName() plugin.TypedName { return p.name }

// RegisterCollector adds a Collector to be polled at the given frequency.
// Safe to call before Start.
func (p *PollingSource) RegisterCollector(c dlsrc.Collector, frequency time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.collectors = append(p.collectors, collectorEntry{collector: c, frequency: frequency})
}

// Start launches one polling goroutine per registered Collector. Returns an error if called more than once.
func (p *PollingSource) Start(ctx context.Context) error {
	if !p.started.CompareAndSwap(false, true) {
		return errors.New("PollingSource already started")
	}
	ctx, p.cancel = context.WithCancel(ctx)

	p.mu.Lock()
	snapshot := make([]collectorEntry, len(p.collectors))
	copy(snapshot, p.collectors)
	p.mu.Unlock()

	for _, entry := range snapshot {
		p.wg.Add(1)
		go p.runCollector(ctx, entry.collector, entry.frequency)
	}
	return nil
}

// Stop cancels all polling goroutines and waits for them to exit.
func (p *PollingSource) Stop() {
	if p.cancel != nil {
		p.cancel()
		p.wg.Wait()
	}
}

func (p *PollingSource) runCollector(ctx context.Context, c dlsrc.Collector, freq time.Duration) {
	defer p.wg.Done()
	logger := log.FromContext(ctx).WithName("polling-source")

	ticker := time.NewTicker(freq)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if _, err := c.Poll(ctx); err != nil {
				logger.Error(err, "collector error", "collector", c.TypedName())
			}
		}
	}
}
