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

// BlockHash is a hash of a block of request data.
type BlockHash uint64

// ModelID is a unique identifier for a model.
type ModelID string

func (m ModelID) String() string {
	return string(m)
}

// PrefixModel contains information about a specific model and its cache capacity.
type PrefixModel struct {
	ModelID
	NumOfPrefixBlocks int
}

// ModelSet holds a set of models that may have a specific prefix hash.
type ModelSet map[ModelID]struct{}

// IndexerInterface maintains an LRU cache of prompt prefix hashes and the model(s) that might have that
// prefix cached.
type IndexerInterface interface {
	Get(hash BlockHash) ModelSet
	Add(hashes []BlockHash, model PrefixModel)
	RemoveModel(modelID ModelID)
	Models() []ModelID
}

// Made with Bob
