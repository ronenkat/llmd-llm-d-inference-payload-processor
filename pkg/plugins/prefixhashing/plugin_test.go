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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/datastore"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/datalayer"
)

// createTestRequest creates a test inference request with the given prompt.
func createTestRequest(prompt string) *framework.InferenceRequest {
	req := framework.NewInferenceRequest()
	req.Body["messages"] = []map[string]interface{}{
		{
			"role":    "user",
			"content": prompt,
		},
	}
	req.Body["model"] = "test-model"
	return req
}

// TestProcessRequest_NoMatching tests ProcessRequest when no hashes match in the indexer.
func TestProcessRequest_NoMatching(t *testing.T) {
	ctx := context.Background()

	// Setup datastore with indexer
	ds := datastore.NewStore()
	// Get or create the indexer
	if ds.GetPrefixIndexer() == nil {
		ds.SetPrefixIndexer(newIndexer(ctx, defaultLRUCapacityPerModel, nil))
	}

	config := Config{
		BlockSizeTokens:        16,
		MaxPrefixBlocksToMatch: 10,
		MaxPrefixTokensToMatch: 0,
		LRUCapacityPerModel:    defaultLRUCapacityPerModel,
	}
	plugin := NewPrefixHashing(ds, config)
	cycleState := framework.NewCycleState()

	// Create request with sufficient content to generate hashes
	request := createTestRequest("This is a test prompt that is long enough to generate prefix hashes for testing purposes.")

	// Process request - indexer is empty, so no matches expected
	err := plugin.ProcessRequest(ctx, cycleState, request)
	require.NoError(t, err)

	// Verify state was written
	state, err := framework.ReadCycleStateKey[*RequestHashingState](cycleState, PrefixHashingPluginType)
	require.NoError(t, err)
	require.NotNil(t, state)

	// Verify hashes were generated
	assert.NotEmpty(t, state.PrefixHashes, "Expected prefix hashes to be generated")

	// Verify no models matched (empty map)
	assert.Empty(t, state.PrefixCacheModels, "Expected no models to match when indexer has no cached hashes")
}

// TestProcessRequest_PartialMatching tests ProcessRequest with partial hash matching.
func TestProcessRequest_PartialMatching(t *testing.T) {
	ctx := context.Background()

	// Setup datastore with indexer
	ds := datastore.NewStore()
	// Get or create the indexer
	idx := ds.GetPrefixIndexer()
	if idx == nil {
		idx = ds.SetPrefixIndexer(newIndexer(ctx, defaultLRUCapacityPerModel, nil))
	}

	config := Config{
		BlockSizeTokens:        16,
		MaxPrefixBlocksToMatch: 10,
		MaxPrefixTokensToMatch: 0,
		LRUCapacityPerModel:    defaultLRUCapacityPerModel,
	}
	plugin := NewPrefixHashing(ds, config)

	// First request - process to populate indexer with hashes for model1
	cycleState1 := framework.NewCycleState()
	request1 := createTestRequest("This is a test prompt that is long enough to generate multiple prefix hashes for testing purposes and partial matching.")

	err := plugin.ProcessRequest(ctx, cycleState1, request1)
	require.NoError(t, err)

	state1, err := framework.ReadCycleStateKey[*RequestHashingState](cycleState1, PrefixHashingPluginType)
	require.NoError(t, err)
	require.NotEmpty(t, state1.PrefixHashes, "Expected hashes from first request")

	// Manually add first hash to indexer for model1, and first two hashes for model2
	if len(state1.PrefixHashes) >= 2 {
		idx.Add([]datalayer.BlockHash{state1.PrefixHashes[0]}, datalayer.PrefixModel{ModelID: datalayer.ModelID("model1"), NumOfPrefixBlocks: 100})
		idx.Add(state1.PrefixHashes[:2], datalayer.PrefixModel{ModelID: datalayer.ModelID("model2"), NumOfPrefixBlocks: 100})
	}

	// Second request - same prompt, should match partially
	cycleState2 := framework.NewCycleState()
	request2 := createTestRequest("This is a test prompt that is long enough to generate multiple prefix hashes for testing purposes and partial matching.")

	err = plugin.ProcessRequest(ctx, cycleState2, request2)
	require.NoError(t, err)

	state2, err := framework.ReadCycleStateKey[*RequestHashingState](cycleState2, PrefixHashingPluginType)
	require.NoError(t, err)
	require.NotNil(t, state2)

	// Verify partial matching:
	// - model1 should have 1 block matched (only first hash)
	// - model2 should have 2 blocks matched (first two hashes)
	assert.Len(t, state2.PrefixCacheModels, 2, "Expected 2 models to have partial matches")
	assert.Equal(t, 1, state2.PrefixCacheModels[datalayer.ModelID("model1")], "Expected model1 to match 1 block")
	assert.Equal(t, 2, state2.PrefixCacheModels[datalayer.ModelID("model2")], "Expected model2 to match 2 blocks")
}

// TestProcessRequest_FullMatching tests ProcessRequest with full hash matching.
func TestProcessRequest_FullMatching(t *testing.T) {
	ctx := context.Background()

	// Setup datastore with indexer
	ds := datastore.NewStore()
	// Get or create the indexer
	idx := ds.GetPrefixIndexer()
	if idx == nil {
		idx = ds.SetPrefixIndexer(newIndexer(ctx, defaultLRUCapacityPerModel, nil))
	}

	config := Config{
		BlockSizeTokens:        16,
		MaxPrefixBlocksToMatch: 10,
		MaxPrefixTokensToMatch: 0,
		LRUCapacityPerModel:    defaultLRUCapacityPerModel,
	}
	plugin := NewPrefixHashing(ds, config)

	// First request - process to populate indexer
	cycleState1 := framework.NewCycleState()
	request1 := createTestRequest("This is a test prompt that is long enough to generate multiple prefix hashes for full matching test.")

	err := plugin.ProcessRequest(ctx, cycleState1, request1)
	require.NoError(t, err)

	state1, err := framework.ReadCycleStateKey[*RequestHashingState](cycleState1, PrefixHashingPluginType)
	require.NoError(t, err)
	require.NotEmpty(t, state1.PrefixHashes, "Expected hashes from first request")

	// Add all hashes to indexer for multiple models
	idx.Add(state1.PrefixHashes, datalayer.PrefixModel{ModelID: datalayer.ModelID("model1"), NumOfPrefixBlocks: 100})
	idx.Add(state1.PrefixHashes, datalayer.PrefixModel{ModelID: datalayer.ModelID("model2"), NumOfPrefixBlocks: 100})
	idx.Add(state1.PrefixHashes, datalayer.PrefixModel{ModelID: datalayer.ModelID("model3"), NumOfPrefixBlocks: 100})

	// Second request - same prompt, should match fully
	cycleState2 := framework.NewCycleState()
	request2 := createTestRequest("This is a test prompt that is long enough to generate multiple prefix hashes for full matching test.")

	err = plugin.ProcessRequest(ctx, cycleState2, request2)
	require.NoError(t, err)

	state2, err := framework.ReadCycleStateKey[*RequestHashingState](cycleState2, PrefixHashingPluginType)
	require.NoError(t, err)
	require.NotNil(t, state2)

	// Verify full matching - all models should match all blocks
	expectedBlockCount := len(state1.PrefixHashes)
	assert.Len(t, state2.PrefixCacheModels, 3, "Expected 3 models to have full matches")
	assert.Equal(t, expectedBlockCount, state2.PrefixCacheModels[datalayer.ModelID("model1")], "Expected model1 to match all blocks")
	assert.Equal(t, expectedBlockCount, state2.PrefixCacheModels[datalayer.ModelID("model2")], "Expected model2 to match all blocks")
	assert.Equal(t, expectedBlockCount, state2.PrefixCacheModels[datalayer.ModelID("model3")], "Expected model3 to match all blocks")
}

// Made with Bob
