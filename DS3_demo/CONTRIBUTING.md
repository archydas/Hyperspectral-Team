# Contributing
- Branch from `main`: `git checkout -b feat/<short-name>`; open a PR; CI must be green.
- `pytest` before pushing. New reader = new test (synthetic file generated in `tests/` or `sampledata.py`).
- Never commit data (`data/raw`, `*.db` are git-ignored) or `.env`. Contract data is confidential — keep the repo **private**.
- Don't change the output contract (docs/CONTRACT.md) without telling Archy and Avinash — they build on it.
- Errors: raise a specific `IngestError` subclass; never guess silently (note guesses in `metadata["notes"]`).
