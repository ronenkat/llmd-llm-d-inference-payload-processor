# Model Config Datasource

Populates the datastore with model names read from a JSON config file and keeps it in sync as the file changes.

It is registered as type `model-config-datasource` and runs as a datalayer datasource.

## What it does

1. Reads a JSON file listing model names (`modelsPath`).
2. Registers each named model in the datastore via `GetOrCreateModel`.
3. Removes any model from the datastore that no longer appears in the file.
4. Watches the file's **parent directory** for `Write`, `Create`, `Remove`, and `Rename` events, re-syncing on every relevant change.

The directory is watched rather than the file directly to handle atomic rename-based replacements, such as Kubernetes ConfigMap remounts.

## Behavioral Intent

This datasource is the authoritative source of which models the system knows about. On every file change the datastore converges to exactly the set of models listed in the file — new models are added and stale models are deleted. Models with an empty `name` field are skipped with a warning log.

## Config file format

```json
{
  "models": [
    { "name": "model-a" },
    { "name": "model-b" }
  ]
}
```

## Configuration

| Field        | Type   | Required | Description                               |
|--------------|--------|----------|-------------------------------------------|
| `modelsPath` | string | yes      | Path to the JSON models config file. Resolved to an absolute path at startup. Must be a file, not a directory. |

### Example plugin config

```json
{
  "type": "model-config-datasource",
  "name": "my-model-config",
  "config": {
    "modelsPath": "/etc/llm-d/models.json"
  }
}
```

## Inputs consumed

- The JSON file at `modelsPath` on disk.

## Outputs produced

- Models registered in (or removed from) the shared `Datastore`.
