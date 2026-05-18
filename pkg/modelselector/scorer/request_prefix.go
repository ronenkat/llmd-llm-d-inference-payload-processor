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

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/datalayer"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/modelselector"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/plugins/prefixhashing"
)

const (
	// RequestPrefixScorerType is the type name for the request prefix scorer plugin.
	RequestPrefixScorerType = "RequestPrefixScorer"
	// PrefixHashingPluginType is the key used to retrieve prefix data from cycle state.
	ConsumesPrefixHashing = "prefix-hashing"
)

var (
	_ modelselector.Scorer = &RequestPrefixScorer{}
)

// RequestPrefixScorer is a scorer plugin that scores models based on their prefix cache match.
// Models with longer prefix matches receive higher scores.
type RequestPrefixScorer struct {
	typedName framework.TypedName
}

// NewRequestPrefixScorer creates a new RequestPrefixScorer instance.
func NewRequestPrefixScorer() *RequestPrefixScorer {
	return &RequestPrefixScorer{
		typedName: framework.TypedName{
			Type: RequestPrefixScorerType,
			Name: RequestPrefixScorerType,
		},
	}
}

// TypedName returns the type and name of the plugin.
func (s *RequestPrefixScorer) TypedName() framework.TypedName {
	return s.typedName
}

// Score scores models based on their prefix cache match ratio.
// The score is calculated as: matchedBlocks / totalBlocks
// - A score of 1.0 means all prefix blocks match (perfect cache hit)
// - A score of 0.0 means no prefix blocks match (cache miss)
// - Scores are normalized to the range [0, 1]
func (s *RequestPrefixScorer) Score(ctx context.Context, cycleState *framework.CycleState, request *framework.InferenceRequest, models []datalayer.Model) map[datalayer.Model]float64 {
	logger := log.FromContext(ctx)
	scores := make(map[datalayer.Model]float64, len(models))

	// Retrieve the prefix state from cycle state
	state, err := framework.ReadCycleStateKey[*prefixhashing.RequestHashingState](cycleState, ConsumesPrefixHashing)
	if err != nil {
		// If prefix data is not available, return zero scores for all models
		logger.V(logging.DEBUG).Info("Prefix state not found in cycle state, returning zero scores", "error", err)
		for _, model := range models {
			scores[model] = 0.0
		}
		return scores
	}

	totalBlocks := len(state.PrefixHashes)
	if totalBlocks == 0 {
		// No prefix hashes available, return zero scores
		logger.V(logging.DEBUG).Info("No prefix hashes available, returning zero scores")
		for _, model := range models {
			scores[model] = 0.0
		}
		return scores
	}

	// Score each model based on its prefix match ratio
	for _, model := range models {
		modelName := model.GetName()
		matchedBlocks := state.PrefixCacheModels[datalayer.ModelID(modelName)]

		// Calculate score as the ratio of matched blocks to total blocks
		score := float64(matchedBlocks) / float64(totalBlocks)

		scores[model] = score

		logger.V(logging.TRACE).Info("Scored model based on prefix match",
			"model", modelName,
			"matchedBlocks", matchedBlocks,
			"totalBlocks", totalBlocks,
			"score", score)
	}

	logger.V(logging.DEBUG).Info("Completed prefix-based scoring",
		"numModels", len(models),
		"totalBlocks", totalBlocks)

	return scores
}

// Made with Bob
