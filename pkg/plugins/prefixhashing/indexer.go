/*
Copyright 2025 The Kubernetes Authors.

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

package prefixhashing

import (
	"context"
	"sync"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-inference-payload-processor/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/datastore"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/datalayer"
)

// indexer implements the IndexerInterface interface.
type indexer struct {
	mu             sync.RWMutex
	hashToModels   map[datalayer.BlockHash]datalayer.ModelSet                      // the lookup data structure to find models that have the BlockHash cached
	modelToLRU     map[datalayer.ModelID]*lru.Cache[datalayer.BlockHash, struct{}] // key is model identifier, value is an LRU cache
	defaultLRUSize int
}

// newIndexer initializes an indexer with size limits and starts cache size reporting.
// If datastore is provided (not nil), it also starts a background cleanup goroutine.
func newIndexer(ctx context.Context, defaultLRUSize int, ds datastore.Datastore) datalayer.IndexerInterface {
	i := &indexer{
		hashToModels:   make(map[datalayer.BlockHash]datalayer.ModelSet),
		modelToLRU:     make(map[datalayer.ModelID]*lru.Cache[datalayer.BlockHash, struct{}]),
		defaultLRUSize: defaultLRUSize,
	}

	// Start cleanup goroutine if datastore is provided
	if ds != nil {
		go i.startCleanup(ctx, ds)
	}

	return i
}

// Add adds a list of prefix hashes to the cache, tied to the model.
func (i *indexer) Add(hashes []datalayer.BlockHash, mdl datalayer.PrefixModel) {
	i.mu.Lock()
	defer i.mu.Unlock()

	// Check if the LRU for this model exists
	lruForModel, exists := i.modelToLRU[mdl.ModelID]
	if !exists {
		lruSize := mdl.NumOfPrefixBlocks
		if lruSize <= 0 {
			lruSize = i.defaultLRUSize
		}
		// We ignore the error since the only possible error is if size <= 0.
		newLRU, _ := lru.NewWithEvict(lruSize, i.makeEvictionFn(mdl.ModelID))
		i.modelToLRU[mdl.ModelID] = newLRU
		lruForModel = newLRU
	}

	// Add to LRU (may evict)
	for _, hash := range hashes {
		lruForModel.Add(hash, struct{}{})
	}

	// Update hashToModels
	for _, hash := range hashes {
		modelIDs := i.hashToModels[hash]
		if modelIDs == nil {
			modelIDs = make(datalayer.ModelSet)
		}
		modelIDs[mdl.ModelID] = struct{}{}
		i.hashToModels[hash] = modelIDs
	}
}

// Get returns a set of models that have the given prefix hash cached.
func (i *indexer) Get(hash datalayer.BlockHash) datalayer.ModelSet {
	i.mu.RLock()
	defer i.mu.RUnlock()

	models := i.hashToModels[hash]
	if models == nil {
		return nil
	}

	res := make(datalayer.ModelSet, len(models))
	for model := range models {
		// Deep copy to avoid race condition.
		res[model] = struct{}{}
	}

	return res
}

// makeEvictionFn returns a per-model LRU eviction callback that removes the model from hashToModels on eviction.
func (i *indexer) makeEvictionFn(modelID datalayer.ModelID) func(datalayer.BlockHash, struct{}) {
	return func(hash datalayer.BlockHash, _ struct{}) {
		// Remove the model from the hash→models map
		if modelSet, ok := i.hashToModels[hash]; ok {
			delete(modelSet, modelID)
			if len(modelSet) == 0 {
				delete(i.hashToModels, hash)
			}
		}
	}
}

// RemoveModel removes a model and its associated entries from the indexer.
func (i *indexer) RemoveModel(modelID datalayer.ModelID) {
	i.mu.Lock()
	defer i.mu.Unlock()

	lruCache, exists := i.modelToLRU[modelID]
	if !exists {
		return
	}

	// Remove all hashes associated with the model from hashToModels (triggers eviction callbacks).
	for _, hash := range lruCache.Keys() {
		lruCache.Remove(hash)
	}

	delete(i.modelToLRU, modelID)
}

// Models returns the list of all models currently tracked in the indexer.
func (i *indexer) Models() []datalayer.ModelID {
	i.mu.RLock()
	defer i.mu.RUnlock()

	models := make([]datalayer.ModelID, 0, len(i.modelToLRU))
	for modelID := range i.modelToLRU {
		models = append(models, modelID)
	}
	return models
}

// startCleanup starts a goroutine that periodically scan the list of models in the datastore
// and removes inactive models from the indexer.
func (i *indexer) startCleanup(ctx context.Context, ds datastore.Datastore) {
	ticker := time.NewTicker(podActiveCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			activeModelSet := make(map[datalayer.ModelID]struct{})
			for _, modelName := range ds.Models() {
				activeModelSet[datalayer.ModelID(modelName)] = struct{}{}
			}

			for _, modelID := range i.Models() {
				if _, ok := activeModelSet[modelID]; !ok {
					i.RemoveModel(modelID)
					log.FromContext(ctx).V(logutil.VERBOSE).Info("Removed model not in active set", "model", modelID)
				}
			}
		}
	}
}

// Made with Bob
