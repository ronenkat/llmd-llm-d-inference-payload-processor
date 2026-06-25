# Request Metadata Extractor

Tracks in-flight request counts and token sums per model and writes them to the datastore on every event batch.

It is registered as type `request-metadata-extractor` and runs as a datalayer extractor.

## What it does

1. Receives a batch of `Event` values from the notification source event loop.
2. On a `RequestEventType` event: increments the request count and adds `max_tokens` to the token sum for the model named in the request body.
3. On a `ResponseEventType` event: decrements the request count and token sum for the same model (floored at zero).
4. Writes the updated `RequestMetadataCount` to each affected model's attribute store under the key `request-metadata`.

## Behavioral Intent

This extractor gives downstream plugins a live view of how many requests are currently in-flight for each model and how many tokens those requests represent. Scorers can use this data to steer traffic away from overloaded models.

**Known limitation:** counters will drift if a request ends without a corresponding `ResponseEventType` (e.g. connection drop, upstream error, context cancellation). The call site is expected to fire a synthetic `ResponseEventType` in its error/EOF path to keep counts accurate.

**Concurrency:** `Extract` is assumed to be called from a single goroutine (the notification source event loop). If parallel dispatch is introduced, a mutex around the counters and datastore write must be added.

## Attribute written

| Key                | Type                   | Description                                        |
|--------------------|------------------------|----------------------------------------------------|
| `request-metadata` | `RequestMetadataCount` | In-flight request count and token sum for a model. |

`RequestMetadataCount` fields:

| Field      | Type    | Description                                       |
|------------|---------|---------------------------------------------------|
| `Requests` | `int64` | Number of requests currently in-flight.           |
| `Tokens`   | `int64` | Sum of `max_tokens` across in-flight requests.    |

## Inputs consumed

- `RequestEventType` events: reads `model` and `max_tokens` from the request body.
- `ResponseEventType` events: reads `model` and `max_tokens` from the request body to reverse the earlier increment.

## Outputs produced

- `request-metadata` attribute updated on the matching model entry in the shared `Datastore`.

## Configuration

The plugin has no config.
