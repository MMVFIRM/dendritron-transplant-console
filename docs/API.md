# Local API

The server is intentionally small and exposes four endpoints.

## `GET /api/health`

Returns readiness without running inference.

## `GET /api/status`

Returns donor provenance, recipient geometry, memory geometry, control modes,
locked examples, environment versions, and the claims boundary.

## `GET /api/benchmark`

Returns the retained five-seed Gate 2C-VM32 benchmark summary.

## `POST /api/analyze`

Example request:

```json
{
  "example_id": "true_if_the",
  "mode": "correct",
  "max_new_tokens": 1
}
```

Custom prompt request:

```json
{
  "prompt": "true if",
  "target": "the",
  "mode": "semantic_opposite",
  "max_new_tokens": 1
}
```

## `POST /api/compare`

```json
{
  "example_id": "true_if_the"
}
```

The comparison is a fixed-model inference intervention. No training occurs.
