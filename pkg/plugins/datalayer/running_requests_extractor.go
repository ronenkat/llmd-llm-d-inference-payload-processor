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
	"sync"
	"sync/atomic"

	fwdatalayer "github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/datalayer"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework"
)

// RunningRequestsCount holds in-flight request and token counts for one model.
type RunningRequestsCount struct {
	Requests int64
	Tokens   int64
}

func (r RunningRequestsCount) Clone() fwdatalayer.Cloneable { return r }

type modelCounters struct {
	requests atomic.Int64
	tokens   atomic.Int64
}

// RunningRequestsExtractor tracks in-flight request counts and token sums per model.
// It writes RunningRequestsCount to each model's "running-requests" attribute.
//
// TODO: counters leak if a request fails without a corresponding ResponseEventType (e.g. connection
// drop, upstream error, context cancellation). The call site should fire a
// synthetic ResponseEventType in its error/EOF path to keep counts accurate.
type RunningRequestsExtractor struct {
	name     framework.TypedName
	handle   framework.Handle
	counters sync.Map // model name -> *modelCounters
}

func NewRunningRequestsExtractor(handle framework.Handle) *RunningRequestsExtractor {
	return &RunningRequestsExtractor{
		name:   framework.TypedName{Type: "RunningRequestsExtractor", Name: "running-requests-extractor"},
		handle: handle,
	}
}

func (e *RunningRequestsExtractor) TypedName() framework.TypedName { return e.name }

func (e *RunningRequestsExtractor) Extract(_ context.Context, events []framework.Event) error {
	touched := make(map[string]struct{})

	for _, ev := range events {
		switch ev.Type {
		case framework.RequestEventType:
			p, ok := ev.Payload.(framework.RequestPayload)
			if !ok {
				continue
			}
			model, _ := p.Request.Body["model"].(string)
			if model == "" {
				continue
			}
			maxTokens, _ := p.Request.Body["max_tokens"].(float64)
			c := e.getOrCreateCounters(model)
			c.requests.Add(1)
			c.tokens.Add(int64(maxTokens))
			touched[model] = struct{}{}

		case framework.ResponseEventType:
			p, ok := ev.Payload.(framework.ResponsePayload)
			if !ok {
				continue
			}
			model, _ := p.Request.Body["model"].(string)
			if model == "" {
				continue
			}
			maxTokens, _ := p.Request.Body["max_tokens"].(float64)
			c := e.getOrCreateCounters(model)
			floorDecrement(&c.requests, 1)
			floorDecrement(&c.tokens, int64(maxTokens))
			touched[model] = struct{}{}
		}
	}

	for model := range touched {
		v, _ := e.counters.Load(model)
		c := v.(*modelCounters)
		e.handle.DataStore().GetOrCreateModel(model).GetAttributes().Put("running-requests", RunningRequestsCount{
			Requests: c.requests.Load(),
			Tokens:   c.tokens.Load(),
		})
	}

	return nil
}

func (e *RunningRequestsExtractor) getOrCreateCounters(model string) *modelCounters {
	v, _ := e.counters.LoadOrStore(model, &modelCounters{})
	return v.(*modelCounters)
}

// floorDecrement decrements counter by delta, flooring at zero atomically.
func floorDecrement(counter *atomic.Int64, delta int64) {
	for {
		old := counter.Load()
		newVal := old - delta
		if newVal < 0 {
			newVal = 0
		}
		if counter.CompareAndSwap(old, newVal) {
			return
		}
	}
}
