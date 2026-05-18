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
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"

	"github.com/cespare/xxhash/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/datalayer"
)

// HashPrompt divides the prompt into blocks and calculates a prefix cache hash for each block.
// The first block hash includes the model name and cache salt (if provided).
// For subsequent blocks, the hash is calculated as: hash(block i content, hash(i-1)).
func HashPrompt(ctx context.Context, request *framework.InferenceRequest, blockSizeTokens int, maxPrefixBlocks int) []datalayer.BlockHash {
	loggerDebug := log.FromContext(ctx).V(logging.DEBUG)
	if request == nil || request.Body == nil {
		loggerDebug.Info("Request or request data is nil, skipping hashing")
		return nil
	}

	userInput, err := getUserInputBytes(request)
	if err != nil {
		loggerDebug.Error(err, "Failed to get user input bytes")
		return nil
	}

	// convert block size from tokens to characters
	cacheBlockSizeChars := blockSizeTokens * averageCharactersPerToken

	if cacheBlockSizeChars <= 0 {
		loggerDebug.Info("Skipping prefix hashing: block size in characters must be positive",
			"blockSizeTokens", blockSizeTokens,
			"cacheBlockSizeChars", cacheBlockSizeChars)
		return nil
	}

	if len(userInput) < cacheBlockSizeChars {
		loggerDebug.Info("Request body too small for prefix cache", "size", len(userInput), "block size in chars", cacheBlockSizeChars)
		return nil
	}

	if len(userInput) > cacheBlockSizeChars*maxPrefixBlocks {
		loggerDebug.Info("Truncating input", "size", len(userInput), "max prefix blocks", maxPrefixBlocks, "block size in chars", cacheBlockSizeChars)
		userInput = userInput[:maxPrefixBlocks*cacheBlockSizeChars]
	}

	// Split the body into blocks of size cacheBlockSizeChars.
	res := make([]datalayer.BlockHash, 0, len(userInput)/cacheBlockSizeChars)

	h := xxhash.New()
	// Different models should have different hashes even with the same body.
	// Different models should have different hashes even with the same body.
	if targetModel, ok := request.Body["model"].(string); ok {
		_, _ = h.Write([]byte(targetModel))
	}
	if cacheSalt, ok := request.Body["cache_salt"].(string); ok && cacheSalt != "" {
		_, _ = h.Write([]byte(cacheSalt))
	}

	prevBlockHash := datalayer.BlockHash(h.Sum64())
	i := 0
	for ; i+cacheBlockSizeChars <= len(userInput); i += cacheBlockSizeChars {
		h.Reset()
		_, _ = h.Write(userInput[i : i+cacheBlockSizeChars])
		_, _ = h.Write(toBytes(prevBlockHash))
		res = append(res, datalayer.BlockHash(h.Sum64()))

		prevBlockHash = res[len(res)-1]
	}

	// 2. Process any remaining bytes as a partial block
	if i < len(userInput) {
		h.Reset()

		_, _ = h.Write(userInput[i:])
		_, _ = h.Write(toBytes(prevBlockHash))
		res = append(res, datalayer.BlockHash(h.Sum64()))
	}

	return res
}

func toBytes(i datalayer.BlockHash) []byte {
	bytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(bytes, uint64(i))
	return bytes
}

func getUserInputBytes(request *framework.InferenceRequest) ([]byte, error) {
	if request.Body == nil {
		return nil, errors.New("request body is nil")
	}

	// Check for Conversations API
	if items, ok := request.Body["items"]; ok {
		return json.Marshal(items)
	}

	// Check for Responses API
	if _, hasInstructions := request.Body["instructions"]; hasInstructions {
		var combined []map[string]interface{}
		if instructions := request.Body["instructions"]; instructions != nil {
			combined = append(combined, map[string]interface{}{"instructions": instructions})
		}
		if tools := request.Body["tools"]; tools != nil {
			combined = append(combined, map[string]interface{}{"tools": tools})
		}
		if input := request.Body["input"]; input != nil {
			combined = append(combined, map[string]interface{}{"input": input})
		}
		return json.Marshal(combined)
	}

	// Check for ChatCompletions API
	if messages, ok := request.Body["messages"]; ok {
		return json.Marshal(messages)
	}

	// Check for Completions API
	if prompt, ok := request.Body["prompt"]; ok {
		return json.Marshal(prompt)
	}

	// Check for Embeddings API
	if input, ok := request.Body["input"]; ok {
		return json.Marshal(input)
	}

	return nil, errors.New("invalid request body: no recognized API format found")
}

// Made with Bob
