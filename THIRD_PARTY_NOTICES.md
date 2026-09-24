# Third-party notices

## Qwen2.5-0.5B-derived numerical assets

The phrase vectors, definition vectors, provenance record, and transfer-card
coefficients in `assets/` were derived offline from the official
`Qwen/Qwen2.5-0.5B` base checkpoint. The checkpoint itself is **not** included
and is never loaded by this demo.

The official model repository identifies Qwen2.5-0.5B as Apache-2.0 licensed.
A copy of the Apache License 2.0 is retained at
`licenses/APACHE-2.0-QWEN.txt`.

Model provenance, pinned revision, source weight SHA-256, extraction layers,
and software/hardware details are retained in
`assets/official_qwen_provenance.json`.

## CPython-derived definition text

The frozen definition-bank metadata contains a deterministic selection of
CPython 3.13 standard-library documentation strings. Python is distributed
under the Python Software Foundation License Version 2 together with
additional historical and incorporated-component notices.

The v1.1.0 recipient weights were additionally trained on non-validation
CPython 3.13 standard-library documentation strings. That text is not
redistributed in this package.

The applicable Python license text is retained at
`licenses/PSF-LICENSE.txt`.

## Original code

The original demo, recipient, Dendritron, VIVERE/MACSL, routing, and web-console
code in this repository is not offered under an open-source license. The
third-party licenses above apply only to their respective derived assets and
components.
