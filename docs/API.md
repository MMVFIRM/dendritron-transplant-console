# Local API

The server is intentionally small and exposes four endpoints.

## `GET /api/health`

Returns readiness without running inference.

## `GET /api/status`

Returns donor provenance, recipient geometry, memory geometry, control modes,
locked examples, environment versions, and the claims boundary.

## `GET /api/benchmark`

Returns the retained five-seed Gate 2C-VM32 phrase-memory summary plus
`recipient_benchmark`: the packaged recipient versus v1.0.0 on the fixed
held-out records.

## `POST /api/analyze`

Example request:

```json
{
  "example_id": "the_path_to",
  "mode": "correct",
  "max_new_tokens": 1
}
```

Custom prompt request:

```json
{
  "prompt": "the path",
  "target": "to",
  "mode": "semantic_opposite",
  "max_new_tokens": 1
}
```

## `POST /api/compare`

```json
{
  "example_id": "the_path_to"
}
```

The comparison is a fixed-model inference intervention. No training occurs.
