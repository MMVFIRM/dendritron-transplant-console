# Uploading this package to GitHub

The release ZIP contains one repository root. Extract it, then run:

```bash
cd dendritron-transplant-console-v1.0.0
git init
git add .
git commit -m "Release Dendritron Transplant Console v1.0.0"
git branch -M main
git remote add origin YOUR_REPOSITORY_REMOTE
git push -u origin main
```

Before pushing, run:

```bash
python scripts/verify_assets.py
python -m pytest -q
```

Recommended repository settings:

- keep GitHub Actions enabled;
- keep the repository private until the live demo and claims language are
  reviewed;
- do not add the original Qwen checkpoint;
- do not remove third-party notices or provenance records;
- preserve the claims boundary in public descriptions;
- enable branch protection after the first clean CI run.

Suggested repository description:

> Live CPU mechanistic demo of donor-state memory transplantation into a sparse
> recurrent Dendritron recipient, with fixed-model corruption controls.

Suggested release title:

```text
Dendritron Transplant Console v1.0.0 — Mechanistic Research Demo
```
