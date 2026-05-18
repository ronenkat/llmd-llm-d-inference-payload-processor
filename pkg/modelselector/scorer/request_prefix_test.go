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

package scorer

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/datalayer"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/plugins/prefixhashing"
)

func TestRequestPrefixScorer_TypedName(t *testing.T) {
	scorer := NewRequestPrefixScorer()
	typedName := scorer.TypedName()

	assert.Equal(t, RequestPrefixScorerType, typedName.Type)
	assert.Equal(t, RequestPrefixScorerType, typedName.Name)
}

func TestRequestPrefixScorer_Score_PerfectMatch(t *testing.T) {
	scorer := NewRequestPrefixScorer()
	cycleState := framework.NewCycleState()
	ctx := context.Background()

	// Create test models
	modelA := datalayer.NewModel("model-a")
	modelB := datalayer.NewModel("model-b")
	models := []datalayer.Model{modelA, modelB}

	// Create prefix state with perfect match for model-a
	state := &prefixhashing.RequestHashingState{
		PrefixHashes: []datalayer.BlockHash{1, 2, 3, 4, 5},
		PrefixCacheModels: map[datalayer.ModelID]int{
			"model-a": 5, // All 5 blocks match
			"model-b": 0, // No blocks match
		},
	}
	cycleState.Write(ConsumesPrefixHashing, state)

	// Score the models
	scores := scorer.Score(ctx, cycleState, framework.NewInferenceRequest(), models)

	// Verify scores
	assert.Equal(t, 1.0, scores[modelA], "model-a should have perfect score of 1.0")
	assert.Equal(t, 0.0, scores[modelB], "model-b should have score of 0.0")
}

func TestRequestPrefixScorer_Score_PartialMatch(t *testing.T) {
	scorer := NewRequestPrefixScorer()
	cycleState := framework.NewCycleState()
	ctx := context.Background()

	// Create test models
	modelA := datalayer.NewModel("model-a")
	modelB := datalayer.NewModel("model-b")
	modelC := datalayer.NewModel("model-c")
	models := []datalayer.Model{modelA, modelB, modelC}

	// Create prefix state with partial matches
	state := &prefixhashing.RequestHashingState{
		PrefixHashes: []datalayer.BlockHash{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		PrefixCacheModels: map[datalayer.ModelID]int{
			"model-a": 8, // 80% match
			"model-b": 5, // 50% match
			"model-c": 2, // 20% match
		},
	}
	cycleState.Write(ConsumesPrefixHashing, state)

	// Score the models
	scores := scorer.Score(ctx, cycleState, framework.NewInferenceRequest(), models)

	// Verify scores
	assert.Equal(t, 0.8, scores[modelA], "model-a should have score of 0.8")
	assert.Equal(t, 0.5, scores[modelB], "model-b should have score of 0.5")
	assert.Equal(t, 0.2, scores[modelC], "model-c should have score of 0.2")
}

func TestRequestPrefixScorer_Score_NoStateInCycleState(t *testing.T) {
	scorer := NewRequestPrefixScorer()
	cycleState := framework.NewCycleState()
	ctx := context.Background()

	// Create test models
	modelA := datalayer.NewModel("model-a")
	modelB := datalayer.NewModel("model-b")
	models := []datalayer.Model{modelA, modelB}

	// Don't write any state to cycle state

	// Score the models
	scores := scorer.Score(ctx, cycleState, framework.NewInferenceRequest(), models)

	// Verify all models get zero scores
	assert.Equal(t, 0.0, scores[modelA], "model-a should have score of 0.0 when no state")
	assert.Equal(t, 0.0, scores[modelB], "model-b should have score of 0.0 when no state")
}

func TestRequestPrefixScorer_Score_EmptyPrefixHashes(t *testing.T) {
	scorer := NewRequestPrefixScorer()
	cycleState := framework.NewCycleState()
	ctx := context.Background()

	// Create test models
	modelA := datalayer.NewModel("model-a")
	modelB := datalayer.NewModel("model-b")
	models := []datalayer.Model{modelA, modelB}

	// Create prefix state with empty hashes
	state := &prefixhashing.RequestHashingState{
		PrefixHashes:      []datalayer.BlockHash{},
		PrefixCacheModels: map[datalayer.ModelID]int{},
	}
	cycleState.Write(ConsumesPrefixHashing, state)

	// Score the models
	scores := scorer.Score(ctx, cycleState, framework.NewInferenceRequest(), models)

	// Verify all models get zero scores
	assert.Equal(t, 0.0, scores[modelA], "model-a should have score of 0.0 with empty hashes")
	assert.Equal(t, 0.0, scores[modelB], "model-b should have score of 0.0 with empty hashes")
}

func TestRequestPrefixScorer_Score_ModelNotInPrefixCache(t *testing.T) {
	scorer := NewRequestPrefixScorer()
	cycleState := framework.NewCycleState()
	ctx := context.Background()

	// Create test models
	modelA := datalayer.NewModel("model-a")
	modelB := datalayer.NewModel("model-b")
	modelC := datalayer.NewModel("model-c")
	models := []datalayer.Model{modelA, modelB, modelC}

	// Create prefix state where model-c is not in the cache map
	state := &prefixhashing.RequestHashingState{
		PrefixHashes: []datalayer.BlockHash{1, 2, 3, 4},
		PrefixCacheModels: map[datalayer.ModelID]int{
			"model-a": 4,
			"model-b": 2,
			// model-c is not in the map
		},
	}
	cycleState.Write(ConsumesPrefixHashing, state)

	// Score the models
	scores := scorer.Score(ctx, cycleState, framework.NewInferenceRequest(), models)

	// Verify scores
	assert.Equal(t, 1.0, scores[modelA], "model-a should have score of 1.0")
	assert.Equal(t, 0.5, scores[modelB], "model-b should have score of 0.5")
	assert.Equal(t, 0.0, scores[modelC], "model-c should have score of 0.0 when not in cache map")
}

func TestRequestPrefixScorer_Score_SingleBlock(t *testing.T) {
	scorer := NewRequestPrefixScorer()
	cycleState := framework.NewCycleState()
	ctx := context.Background()

	// Create test models
	modelA := datalayer.NewModel("model-a")
	modelB := datalayer.NewModel("model-b")
	models := []datalayer.Model{modelA, modelB}

	// Create prefix state with single block
	state := &prefixhashing.RequestHashingState{
		PrefixHashes: []datalayer.BlockHash{1},
		PrefixCacheModels: map[datalayer.ModelID]int{
			"model-a": 1,
			"model-b": 0,
		},
	}
	cycleState.Write(ConsumesPrefixHashing, state)

	// Score the models
	scores := scorer.Score(ctx, cycleState, framework.NewInferenceRequest(), models)

	// Verify scores
	assert.Equal(t, 1.0, scores[modelA], "model-a should have score of 1.0")
	assert.Equal(t, 0.0, scores[modelB], "model-b should have score of 0.0")
}

func TestRequestPrefixScorer_Score_AllModelsEqualMatch(t *testing.T) {
	scorer := NewRequestPrefixScorer()
	cycleState := framework.NewCycleState()
	ctx := context.Background()

	// Create test models
	modelA := datalayer.NewModel("model-a")
	modelB := datalayer.NewModel("model-b")
	modelC := datalayer.NewModel("model-c")
	models := []datalayer.Model{modelA, modelB, modelC}

	// Create prefix state where all models have same match
	state := &prefixhashing.RequestHashingState{
		PrefixHashes: []datalayer.BlockHash{1, 2, 3, 4, 5},
		PrefixCacheModels: map[datalayer.ModelID]int{
			"model-a": 3,
			"model-b": 3,
			"model-c": 3,
		},
	}
	cycleState.Write(ConsumesPrefixHashing, state)

	// Score the models
	scores := scorer.Score(ctx, cycleState, framework.NewInferenceRequest(), models)

	// Verify all models have same score
	expectedScore := 0.6 // 3/5
	assert.Equal(t, expectedScore, scores[modelA], "model-a should have score of 0.6")
	assert.Equal(t, expectedScore, scores[modelB], "model-b should have score of 0.6")
	assert.Equal(t, expectedScore, scores[modelC], "model-c should have score of 0.6")
}

// Made with Bob
