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

package datastore

import (
	"sync"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/datalayer"
)

// Datastore is the interface for reading and updating the model store.
type Datastore interface {
	GetOrCreateModel(name string) datalayer.Model
	DeleteModel(name string)
	Models() []string
	GetPrefixIndexer() datalayer.IndexerInterface
	SetPrefixIndexer(indexer datalayer.IndexerInterface) datalayer.IndexerInterface
}

// Store is the global datastore instance.
var Store Datastore

var (
	once sync.Once
)

// store is a thread-safe registry of Model entries keyed by model name.
// The outer key is the model name; each Model holds an AttributeMap for
// dynamic runtime metrics (e.g. "running-requests", "pool-latency") and
// any static metadata added in future (e.g. vendor, family).
//
// All operations are thread-safe using RWMutex.
type store struct {
	mu            sync.RWMutex
	models        map[string]datalayer.Model
	prefixIndexer datalayer.IndexerInterface
}

// NewStore creates and returns a new Datastore instance with an initialized prefix indexer.
// When called, it also initializes the global Store variable exactly once using sync.Once.
func NewStore() Datastore {
	once.Do(func() {
		Store = &store{
			models:        make(map[string]datalayer.Model),
			prefixIndexer: nil, // Will be set externally via SetPrefixIndexer
		}
	})
	return Store
}

// SetPrefixIndexer sets the prefix indexer for the store.
// This must be called after creating the store to initialize the indexer.
// If the indexer is already set, it returns the current indexer without modification.
func (s *store) SetPrefixIndexer(indexer datalayer.IndexerInterface) datalayer.IndexerInterface {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.prefixIndexer != nil {
		return s.prefixIndexer
	}
	s.prefixIndexer = indexer
	return indexer
}

// GetPrefixIndexer returns the prefix indexer.
func (s *store) GetPrefixIndexer() datalayer.IndexerInterface {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.prefixIndexer
}

// GetOrCreateModel returns the Model for name, creating it atomically if it does not exist.
func (s *store) GetOrCreateModel(name string) datalayer.Model {
	s.mu.Lock()
	defer s.mu.Unlock()
	if m, ok := s.models[name]; ok {
		return m
	}
	m := datalayer.NewModel(name)
	s.models[name] = m
	return m
}

// DeleteModel removes a model by name. No-op if it does not exist.
func (s *store) DeleteModel(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.models, name)
}

// Models returns the names of all tracked models. Order is not guaranteed.
func (s *store) Models() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	names := make([]string, 0, len(s.models))
	for n := range s.models {
		names = append(names, n)
	}
	return names
}
