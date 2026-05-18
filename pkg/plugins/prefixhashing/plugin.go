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

package prefixhashing

import (
	"context"
	"encoding/json"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/datastore"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/datalayer"
)

const (
	// PrefixHashingPluginType is the identifier used when registering this plugin.
	PrefixHashingPluginType = "prefix-hashing"
)

// compile-time interface assertion
var _ framework.RequestProcessor = &PrefixHashing{}

// PrefixHashingFactory creates a PrefixHashing plugin with default configuration.
func PrefixHashingFactory(name string, configData json.RawMessage, handle framework.Handle) (framework.Plugin, error) {
	config := DefaultConfig
	if len(configData) > 0 {
		if err := json.Unmarshal(configData, &config); err != nil {
			return nil, err
		}
	}

	// Check if the prefix indexer is nil and create one if needed
	if datastore.Store.GetPrefixIndexer() == nil {
		ctx := handle.Context()
		indexer := newIndexer(ctx, config.LRUCapacityPerModel, datastore.Store)
		datastore.Store.SetPrefixIndexer(indexer)
	}

	return NewPrefixHashing(datastore.Store, config).WithName(name), nil
}

// PrefixHashing is a RequestProcessor plugin that extracts prefix hashes from requests
// and stores them in the cycle state for use by the prefix scorer.
type PrefixHashing struct {
	typeName  framework.TypedName
	dataStore datastore.Datastore
	config    Config
}

// NewPrefixHashing creates a new PrefixHashing instance.
func NewPrefixHashing(ds datastore.Datastore, config Config) *PrefixHashing {
	return &PrefixHashing{
		typeName:  framework.TypedName{Type: PrefixHashingPluginType, Name: PrefixHashingPluginType},
		dataStore: ds,
		config:    config,
	}
}

// TypedName returns the type and name of the plugin.
func (p *PrefixHashing) TypedName() framework.TypedName {
	return p.typeName
}

// WithName sets the instance name, used by the factory when the plugin is configured by name.
func (p *PrefixHashing) WithName(name string) *PrefixHashing {
	p.typeName.Name = name
	return p
}

// ProcessRequest implements the RequestProcessor interface.
// It computes prefix hashes for the request and stores them in the cycle state.
func (p *PrefixHashing) ProcessRequest(ctx context.Context, cycleState *framework.CycleState, request *framework.InferenceRequest) error {
	logger := log.FromContext(ctx)

	// Calculate block size and max blocks
	blockSize := p.config.BlockSizeTokens
	maxBlocks := p.config.MaxPrefixBlocksToMatch
	if p.config.MaxPrefixTokensToMatch > 0 && blockSize > 0 {
		maxBlocks = p.config.MaxPrefixTokensToMatch / blockSize
	}

	// Hash the prompt to get prefix hashes
	hashes := HashPrompt(ctx, request, blockSize, maxBlocks)
	if len(hashes) == 0 {
		logger.V(logging.DEBUG).Info("No prefix hashes generated for request")
		return nil
	}

	// Get the indexer from the datastore
	indexer := p.dataStore.GetPrefixIndexer()
	if indexer == nil {
		logger.V(logging.DEBUG).Info("Prefix indexer not available")
		// Store empty state
		state := &RequestHashingState{
			PrefixHashes:      hashes,
			PrefixCacheModels: make(map[datalayer.ModelID]int),
		}
		cycleState.Write(PrefixHashingPluginType, state)
		return nil
	}

	// Match longest prefix for each model
	prefixCacheModels := p.matchLongestPrefix(ctx, hashes, indexer)

	// Store the state in cycle state for use by the scorer
	state := &RequestHashingState{
		PrefixHashes:      hashes,
		PrefixCacheModels: prefixCacheModels,
	}
	cycleState.Write(PrefixHashingPluginType, state)

	logger.V(logging.TRACE).Info("Stored prefix state in cycle state",
		"totalBlocks", len(hashes),
		"blockSize", blockSize,
		"matchedModels", len(prefixCacheModels))

	return nil
}

// matchLongestPrefix returns a map of models and the length of prefix that each model caches.
// The prefix length is defined in blocks.
func (p *PrefixHashing) matchLongestPrefix(ctx context.Context, hashes []datalayer.BlockHash, indexer datalayer.IndexerInterface) map[datalayer.ModelID]int {
	logger := log.FromContext(ctx).V(logging.TRACE)
	res := make(map[datalayer.ModelID]int)

	// Use a greedy strategy to search from the longest prefix
	for _, hash := range hashes {
		cachedModels := indexer.Get(hash)
		if len(cachedModels) == 0 {
			break
		}
		logger.Info("Found cached models", "cachedModels", cachedModels, "total # blocks", len(hashes))
		for modelID := range cachedModels {
			res[modelID]++
		}
	}
	return res
}

// Made with Bob
