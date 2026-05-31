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
	"context"
	"encoding/json"
	"slices"
	"testing"

	ctrlbuilder "sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/datastore"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/interface/datalayer"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/interface/modelselector"
	fwkplugin "github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/plugins/modelselector/picker/maxscore"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/plugins/modelselector/scorer/costaware"
	ms "github.com/llm-d/llm-d-inference-payload-processor/pkg/modelselector"
)

// fakeHandle implements plugin.Handle for unit tests.
type fakeHandle struct {
	ds      datalayer.Datastore
	plugins map[string]fwkplugin.Plugin
}

func (f *fakeHandle) Context() context.Context                { return context.Background() }
func (f *fakeHandle) Client() client.Client                   { return nil }
func (f *fakeHandle) ReconcilerBuilder() *ctrlbuilder.Builder { return nil }
func (f *fakeHandle) Datastore() datalayer.Datastore          { return f.ds }

func (f *fakeHandle) Plugin(name string) fwkplugin.Plugin { return f.plugins[name] }
func (f *fakeHandle) AddPlugin(name string, p fwkplugin.Plugin) {
	f.plugins[name] = p
}
func (f *fakeHandle) GetAllPlugins() []fwkplugin.Plugin {
	result := make([]fwkplugin.Plugin, 0, len(f.plugins))
	for _, p := range f.plugins {
		result = append(result, p)
	}
	return result
}
func (f *fakeHandle) GetAllPluginsWithNames() map[string]fwkplugin.Plugin { return f.plugins }

func newTestDatastore(modelNames ...string) datalayer.Datastore {
	ds := datastore.NewFakeDataStore()
	for _, name := range modelNames {
		ds.GetOrCreateModel(name)
	}
	return ds
}

// newFakeHandle creates a fakeHandle with a datastore pre-populated with the given model names
// and no additional plugins configured.
func newFakeHandle(modelNames ...string) *fakeHandle {
	return &fakeHandle{
		ds:      newTestDatastore(modelNames...),
		plugins: map[string]fwkplugin.Plugin{},
	}
}

// mustFactory calls ModelSelectorPluginFactory and fails the test on error.
func mustFactory(t *testing.T, parameters json.RawMessage, handle *fakeHandle) *ModelSelectorPlugin {
	t.Helper()
	plug, err := ModelSelectorPluginFactory(ModelSelectorPluginType, parameters, handle)
	if err != nil {
		t.Fatalf("ModelSelectorPluginFactory failed: %v", err)
	}
	return plug.(*ModelSelectorPlugin)
}

// profile returns the ModelSelectorProfile for inspection in tests.
func profile(p *ModelSelectorPlugin) *ms.ModelSelectorProfile {
	return p.selector.Profile()
}

// TestProcessRequestSelectsFromDatastoreModels checks that the selected model is one of the candidates registered in the datastore.
func TestProcessRequestSelectsFromDatastoreModels(t *testing.T) {
	candidates := []string{"llama-70b", "llama-8b", "mistral-7b"}
	p := mustFactory(t, json.RawMessage(`{}`), newFakeHandle(candidates...))

	request := requesthandling.NewInferenceRequest()
	request.Body["model"] = "auto"
	cycleState := fwkplugin.NewCycleState()

	if err := p.ProcessRequest(context.Background(), cycleState, request); err != nil {
		t.Fatalf("ProcessRequest failed: %v", err)
	}

	selectedModel := request.Body["model"].(string)
	if !slices.Contains(candidates, selectedModel) {
		t.Errorf("selected model %q is not in datastore models %v", selectedModel, candidates)
	}
}

// TestProcessRequestFailsWithEmptyDatastore checks that ProcessRequest returns an error when no candidate models are available.
func TestProcessRequestFailsWithEmptyDatastore(t *testing.T) {
	p := mustFactory(t, json.RawMessage(`{}`), newFakeHandle())

	request := requesthandling.NewInferenceRequest()
	request.Body["model"] = "auto"
	cycleState := fwkplugin.NewCycleState()

	if err := p.ProcessRequest(context.Background(), cycleState, request); err == nil {
		t.Fatal("expected error with empty datastore")
	}
}

// TestTypedName checks that the plugin's TypedName type matches the registered ModelSelectorPluginType constant.
func TestTypedName(t *testing.T) {
	thePlugin := mustFactory(t, json.RawMessage(`{}`), newFakeHandle("model-a"))
	if thePlugin.TypedName().Type != ModelSelectorPluginType {
		t.Errorf("expected type %q, got %q", ModelSelectorPluginType, thePlugin.TypedName().Type)
	}
}

// TestFactoryUsesDefaultMaxScorePickerWhenNoPluginsConfigured checks that MaxScorePicker is used as the default when plugins list is empty.
func TestFactoryUsesDefaultMaxScorePickerWhenNoPluginsConfigured(t *testing.T) {
	thePlugin := mustFactory(t, json.RawMessage(`{}`), newFakeHandle("model-a"))
	picker := profile(thePlugin).Picker()
	if picker == nil || picker.TypedName().Type != maxscore.MaxScorePickerType {
		t.Errorf("expected default picker type %q, got %v", maxscore.MaxScorePickerType, picker)
	}
}

// TestFactoryWiresScorerFromParameters checks that a scorer plugin referenced in parameters is added to the profile with the given weight.
func TestFactoryWiresScorerFromParameters(t *testing.T) {
	scorer := costaware.NewCostScorer()
	handle := newFakeHandle("model-a", "model-b")
	handle.AddPlugin(scorer.TypedName().Name, scorer)

	p := mustFactory(t, json.RawMessage(`{"plugins":[{"pluginRef":"cost-scorer","weight":2.0}]}`), handle)
	scorers := profile(p).Scorers()
	if len(scorers) != 1 || scorers[0].TypedName().Type != costaware.CostScorerType {
		t.Errorf("expected one scorer of type %q, got %v", costaware.CostScorerType, scorers)
	}
	if scorers[0].Weight() != 2.0 {
		t.Errorf("expected scorer weight 2.0, got %v", scorers[0].Weight())
	}
}

// TestFactoryWiresPickerFromParameters checks that a picker plugin referenced in parameters is used instead of the default.
func TestFactoryWiresPickerFromParameters(t *testing.T) {
	picker := maxscore.NewMaxScorePicker()
	handle := newFakeHandle("model-a")
	handle.AddPlugin(picker.TypedName().Name, picker)

	p := mustFactory(t, json.RawMessage(`{"plugins":[{"pluginRef":"max-score-picker"}]}`), handle)
	got := profile(p).Picker()
	if got == nil || got.TypedName().Type != maxscore.MaxScorePickerType {
		t.Errorf("expected picker type %q, got %v", maxscore.MaxScorePickerType, got)
	}
}

// TestFactoryRejectsMultiplePickers checks that referencing more than one picker plugin in parameters returns an error.
func TestFactoryRejectsMultiplePickers(t *testing.T) {
	p1 := maxscore.NewMaxScorePicker().WithName("picker-1")
	p2 := maxscore.NewMaxScorePicker().WithName("picker-2")
	handle := newFakeHandle("model-a")
	handle.AddPlugin("picker-1", p1)
	handle.AddPlugin("picker-2", p2)

	_, err := ModelSelectorPluginFactory(ModelSelectorPluginType,
		json.RawMessage(`{"plugins":[{"pluginRef":"picker-1"},{"pluginRef":"picker-2"}]}`),
		handle)
	if err == nil {
		t.Fatal("expected error when two picker plugins are configured")
	}
}

// TestFactoryRejectsScorerWithoutWeight checks that a scorer pluginRef without a weight returns an error.
func TestFactoryRejectsScorerWithoutWeight(t *testing.T) {
	scorer := costaware.NewCostScorer()
	handle := newFakeHandle("model-a")
	handle.AddPlugin(scorer.TypedName().Name, scorer)

	_, err := ModelSelectorPluginFactory(ModelSelectorPluginType,
		json.RawMessage(`{"plugins":[{"pluginRef":"cost-scorer"}]}`),
		handle)
	if err == nil {
		t.Fatal("expected error when scorer has no weight")
	}
}

// TestFactoryRejectsUnknownPluginRef checks that referencing a plugin not in the handle returns an error.
func TestFactoryRejectsUnknownPluginRef(t *testing.T) {
	_, err := ModelSelectorPluginFactory(ModelSelectorPluginType,
		json.RawMessage(`{"plugins":[{"pluginRef":"nonexistent-plugin"}]}`),
		newFakeHandle("model-a"))
	if err == nil {
		t.Fatal("expected error for unknown pluginRef")
	}
}

// fakeScorerFilter implements both modelselector.Scorer and modelselector.Filter.
type fakeScorerFilter struct{ typedName fwkplugin.TypedName }

func (f *fakeScorerFilter) TypedName() fwkplugin.TypedName { return f.typedName }
func (f *fakeScorerFilter) Score(_ context.Context, _ *fwkplugin.CycleState, _ *requesthandling.InferenceRequest, models []datalayer.Model) map[datalayer.Model]float64 {
	out := make(map[datalayer.Model]float64, len(models))
	for _, m := range models {
		out[m] = 1.0
	}
	return out
}
func (f *fakeScorerFilter) Filter(_ context.Context, _ *fwkplugin.CycleState, _ *requesthandling.InferenceRequest, models []datalayer.Model) []datalayer.Model {
	return models
}

var _ modelselector.Scorer = &fakeScorerFilter{}
var _ modelselector.Filter = &fakeScorerFilter{}

// TestFactoryPluginImplementingBothScorerAndFilter checks that a plugin implementing both Scorer and Filter is registered in both roles.
func TestFactoryPluginImplementingBothScorerAndFilter(t *testing.T) {
	dual := &fakeScorerFilter{typedName: fwkplugin.TypedName{Type: "dual", Name: "dual"}}
	handle := newFakeHandle("model-a")
	handle.AddPlugin("dual", dual)

	p := mustFactory(t, json.RawMessage(`{"plugins":[{"pluginRef":"dual","weight":1.0}]}`), handle)
	prof := profile(p)
	if len(prof.Filters()) != 1 || prof.Filters()[0].TypedName().Name != "dual" {
		t.Errorf("expected dual in filters, got %v", prof.Filters())
	}
	if len(prof.Scorers()) != 1 || prof.Scorers()[0].TypedName().Name != "dual" {
		t.Errorf("expected dual in scorers, got %v", prof.Scorers())
	}
}
