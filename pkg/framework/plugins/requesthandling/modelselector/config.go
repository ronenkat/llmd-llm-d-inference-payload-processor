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

package modelselector

import (
	"encoding/json"
	"fmt"
)

// PluginRefConfig references a plugin by name and optionally assigns a weight.
// Weight is required when the referenced plugin is a Scorer.
type PluginRefConfig struct {
	PluginRef string   `json:"pluginRef"`
	Weight    *float64 `json:"weight,omitempty"`
}

// ModelSelectorPluginConfig holds the configuration for the ModelSelectorPlugin.
type ModelSelectorPluginConfig struct {
	Plugins []PluginRefConfig `json:"plugins,omitempty"`
}

func parseConfig(parameters json.RawMessage) (*ModelSelectorPluginConfig, error) {
	cfg := &ModelSelectorPluginConfig{}
	if len(parameters) == 0 {
		return cfg, nil
	}
	if err := json.Unmarshal(parameters, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse model-selector config: %w", err)
	}
	return cfg, nil
}
