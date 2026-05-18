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

package prefixindexing

import (
	"context"
	"encoding/json"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/datastore"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/datalayer"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/datalayer/datasource"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/plugins/prefixhashing"
)

const (
	// PluginType is the identifier used when registering this extractor.
	PluginType = "prefix-indexing-extractor"
)

// compile-time interface assertion
var _ datasource.Extractor = &PrefixIndexingExtractor{}

// Factory creates a PrefixIndexingExtractor with the global DataStore and default configuration.
func Factory(name string, configData json.RawMessage, _ framework.Handle) (framework.Plugin, error) {
	config := prefixhashing.DefaultConfig
	if len(configData) > 0 {
		if err := json.Unmarshal(configData, &config); err != nil {
			return nil, err
		}
	}
	return NewPrefixIndexingExtractor(datastore.Store, config).WithName(name), nil
}

// PrefixIndexingExtractor tracks prefix hashes used by models and updates the prefix indexer.
// It processes ResponseEventType events to record which models have cached which prefix hashes.
//
// Extract is assumed to be called from a single goroutine (the NotificationSource event loop).
type PrefixIndexingExtractor struct {
	typeName  framework.TypedName
	dataStore datastore.Datastore
	config    prefixhashing.Config
}

// NewPrefixIndexingExtractor creates a new PrefixIndexingExtractor instance.
func NewPrefixIndexingExtractor(ds datastore.Datastore, config prefixhashing.Config) *PrefixIndexingExtractor {
	return &PrefixIndexingExtractor{
		typeName:  framework.TypedName{Type: PluginType, Name: PluginType},
		dataStore: ds,
		config:    config,
	}
}

// TypedName returns the type and name of the plugin.
func (e *PrefixIndexingExtractor) TypedName() framework.TypedName {
	return e.typeName
}

// WithName sets the instance name, used by the factory when the plugin is configured by name.
func (e *PrefixIndexingExtractor) WithName(name string) *PrefixIndexingExtractor {
	e.typeName.Name = name
	return e
}

// Extract processes events and updates the prefix indexer with prefix hashes used by models.
// It processes ResponseEventType events to track which models have processed requests with
// specific prefix hashes.
func (e *PrefixIndexingExtractor) Extract(ctx context.Context, events []datasource.Event) error {
	logger := log.FromContext(ctx)

	// Get the prefix indexer from the datastore
	indexer := e.dataStore.GetPrefixIndexer()
	if indexer == nil {
		logger.V(logging.DEBUG).Info("Prefix indexer not available, skipping prefix indexing")
		return nil
	}

	// Cast indexer to access the Add method
	concreteIndexer, ok := indexer.(interface {
		Add(hashes []datalayer.BlockHash, model datalayer.PrefixModel)
	})
	if !ok {
		logger.V(logging.DEBUG).Info("Indexer does not support Add method")
		return nil
	}

	for _, ev := range events {
		// We only process response events to track completed requests
		if ev.Type != datasource.ResponseEventType {
			continue
		}

		payload, ok := ev.Payload.(datasource.ResponsePayload)
		if !ok {
			continue
		}

		// Get the model name from the request
		modelName, _ := payload.Request.Body["model"].(string)
		if modelName == "" {
			continue
		}

		// Calculate block size and max blocks
		blockSize := e.config.BlockSizeTokens
		maxBlocks := e.config.MaxPrefixBlocksToMatch
		if e.config.MaxPrefixTokensToMatch > 0 && blockSize > 0 {
			maxBlocks = e.config.MaxPrefixTokensToMatch / blockSize
		}

		// Compute prefix hashes directly from the request
		hashes := prefixhashing.HashPrompt(ctx, payload.Request, blockSize, maxBlocks)
		if len(hashes) == 0 {
			logger.V(logging.DEBUG).Info("No prefix hashes generated for request",
				"model", modelName)
			continue
		}

		// Update the indexer with the hashes for this model
		model := datalayer.PrefixModel{
			ModelID:           datalayer.ModelID(modelName),
			NumOfPrefixBlocks: 0, // Use default from indexer
		}
		concreteIndexer.Add(hashes, model)

		logger.V(logging.TRACE).Info("Updated prefix indexer",
			"model", modelName,
			"numHashes", len(hashes),
			"blockSize", blockSize)
	}

	return nil
}

// Made with Bob
