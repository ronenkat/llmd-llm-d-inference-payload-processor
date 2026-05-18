/*
Copyright 2026 The Kubernetes Authors.

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
	"time"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/datalayer"
)

// RequestHashingState is request hashes and models that have previously seen these hashes
type RequestHashingState struct {
	// PrefixHashes is a list of prefix hashes of the request prompt broken into blocks.
	PrefixHashes []datalayer.BlockHash
	// A map of model to its longest prefix cache match length in blocks.
	PrefixCacheModels map[datalayer.ModelID]int
}

// Clone creates a deep copy of the RequestHashingState.
func (s *RequestHashingState) Clone() *RequestHashingState {
	prefixHashes := make([]datalayer.BlockHash, len(s.PrefixHashes))
	copy(prefixHashes, s.PrefixHashes)
	prefixCacheModels := make(map[datalayer.ModelID]int, len(s.PrefixCacheModels))
	for key, value := range s.PrefixCacheModels {
		prefixCacheModels[key] = value
	}

	return &RequestHashingState{
		PrefixHashes:      prefixHashes,
		PrefixCacheModels: prefixCacheModels,
	}
}

const (
	// podActiveCheckInterval is the interval at which we check if models are still active.
	podActiveCheckInterval = 2 * time.Minute

	// defaultBlockSizeTokens is the default token block size (vLLM default is 16).
	defaultBlockSizeTokens = 16

	// defaultMaxPrefixBlocks is the maximum number of blocks to match.
	// Two long requests with the same prefix up to this limit will be indistinguishable.
	// This parameter provides a trade-off between cache size, prefix matching speed and matching
	// accuracy. Use a small value if most requests are short to reduce cache size and speed up the
	// matching process. Use a large value if most requests are long to increase the matching accuracy.
	defaultMaxPrefixBlocks = 256

	// defaultLRUCapacityPerModel is the default capacity of the LRU indexer per model.
	// The indexer is an approximation to the actual prefix LRU cache state on the model instances per model.
	// A small capacity ensures a high accuracy of cache hit on the model server, but it will
	// increase the chance of false negatives. A high capacity does the opposite.
	// To properly size this, consider the sum of the total number of cache entries on all model
	// servers. Consider the llama3 8B model on a H100 80GB GPUs. The size of the model weight is
	// about 16GB. The remaining HBM used for caching prefixes is 64GB. Each
	// token is about 128KB in size, so we can cache 500K tokens. Using the default block size of 16
	// in vLLM, we will have 250K / 16 = 31.25K blocks.
	defaultLRUCapacityPerModel = 131072

	// averageCharactersPerToken is an estimated average characters per token.
	averageCharactersPerToken = 4
)

// Config defines the configuration for the prefix cache plugins.
type Config struct {
	// The input prompt is broken into sizes of BlockSizeTokens to calculate block hashes.
	BlockSizeTokens int `json:"blockSizeTokens"`
	// Deprecated: Legacy block size defined in number of characters.
	BlockSize int `json:"blockSize"`
	// MaxPrefixBlocksToMatch is the maximum number of prefix blocks to match.
	MaxPrefixBlocksToMatch int `json:"maxPrefixBlocksToMatch"`
	// MaxPrefixTokensToMatch is the maximum number of prefix tokens to match.
	// When set (> 0), it takes precedence over MaxPrefixBlocksToMatch by computing
	// maxBlocks = MaxPrefixTokensToMatch / blockSizeTokens.
	MaxPrefixTokensToMatch int `json:"maxPrefixTokensToMatch"`
	// Max capacity size of the LRU indexer in number of entries per model.
	LRUCapacityPerModel int `json:"lruCapacityPerModel"`
}

// DefaultConfig provides sensible defaults for the prefix cache plugins.
var DefaultConfig = Config{
	BlockSize:              0,
	BlockSizeTokens:        defaultBlockSizeTokens,
	MaxPrefixBlocksToMatch: defaultMaxPrefixBlocks,
	LRUCapacityPerModel:    defaultLRUCapacityPerModel,
}

// Made with Bob
