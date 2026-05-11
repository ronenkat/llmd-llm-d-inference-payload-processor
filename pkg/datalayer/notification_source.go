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

package datalayer

import (
	"context"
	"errors"
	"sync/atomic"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework"
)

const (
	defaultBufferSize   = 10000
	defaultTickInterval = 100 * time.Millisecond
)

type notificationSource struct {
	name       framework.TypedName
	ch         chan framework.Event
	extractors []framework.Extractor
	interval   time.Duration

	started atomic.Bool
	cancel  context.CancelFunc
	done    chan struct{}
}

// NewNotificationSource creates a NotificationSource that fans event batches
// to the given extractors on every tick.
func NewNotificationSource(name string, extractors ...framework.Extractor) framework.NotificationSource {
	return &notificationSource{
		name:       framework.TypedName{Type: "NotificationSource", Name: name},
		ch:         make(chan framework.Event, defaultBufferSize),
		extractors: extractors,
		interval:   defaultTickInterval,
		done:       make(chan struct{}),
	}
}

func (n *notificationSource) TypedName() framework.TypedName { return n.name }

func (n *notificationSource) RegisterExtractor(e framework.Extractor) {
	n.extractors = append(n.extractors, e)
}

// Notify fires an event non-blocking; drops silently if the buffer is full.
func (n *notificationSource) Notify(e framework.Event) {
	select {
	case n.ch <- e:
	default:
	}
}

// Start launches the tick loop. Returns an error if called more than once.
func (n *notificationSource) Start(ctx context.Context) error {
	if !n.started.CompareAndSwap(false, true) {
		return errors.New("NotificationSource already started")
	}
	ctx, n.cancel = context.WithCancel(ctx)
	ready := make(chan struct{})
	go n.tickLoop(ctx, ready)
	<-ready
	return nil
}

// Stop cancels the tick loop and waits for it to exit.
func (n *notificationSource) Stop() {
	if n.cancel != nil {
		n.cancel()
		<-n.done
	}
}

func (n *notificationSource) tickLoop(ctx context.Context, ready chan struct{}) {
	ticker := time.NewTicker(n.interval)
	defer ticker.Stop()
	close(ready)

	logger := log.FromContext(ctx).WithName("notification-source")

	for {
		select {
		case <-ctx.Done():
			close(n.done)
			return
		case <-ticker.C:
			batch := n.drain()
			// Extractors are called sequentially; current extractors are in-memory only.
			// Switch to a WaitGroup if any extractor performs I/O.
			for _, ext := range n.extractors {
				if err := ext.Extract(ctx, batch); err != nil {
					logger.Error(err, "extractor error", "extractor", ext.TypedName())
				}
			}
		}
	}
}

// drain reads all pending events from the channel without blocking.
func (n *notificationSource) drain() []framework.Event {
	var batch []framework.Event
	for {
		select {
		case e := <-n.ch:
			batch = append(batch, e)
		default:
			return batch
		}
	}
}
