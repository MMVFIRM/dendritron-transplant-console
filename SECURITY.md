# Security

## Intended deployment

The server binds to `127.0.0.1` by default and has no authentication. It is
intended for a single-user local demonstration.

Do not expose the server directly to an untrusted network. Binding with
`--host 0.0.0.0` should occur only behind an appropriate authenticated reverse
proxy or inside a controlled demonstration environment.

## Defensive properties

- No donor checkpoint is included or downloaded.
- Recipient checkpoints load through `torch.load(..., weights_only=True)`.
- Packaged assets are SHA-256 verified before launch by the one-click scripts.
- Request bodies are limited to one megabyte.
- Static file paths are resolved beneath the fixed web root.
- A restrictive local Content Security Policy is sent with the frontend.
- No analytics, telemetry, cookies, accounts, or external browser assets are
  used.

## Reporting

Report suspected vulnerabilities privately to the repository owner before
opening a public issue.
