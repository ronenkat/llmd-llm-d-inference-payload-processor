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

package runnable

import (
	"context"

	"sigs.k8s.io/controller-runtime/pkg/manager"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/interface/datalayer/datasource"
)

// DataSourceRunnable wraps a DataSource as a manager.Runnable.
// It starts the data source, waits for context cancellation, then stops it gracefully.
func DataSourceRunnable(ds datasource.DataSource) manager.Runnable {
	return manager.RunnableFunc(func(ctx context.Context) error {
		if err := ds.Start(ctx); err != nil {
			return err
		}
		<-ctx.Done()
		ds.Stop()
		return nil
	})
}
