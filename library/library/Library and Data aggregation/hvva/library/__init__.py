"""Spectral library storage.

PostgreSQL material library (DS3 schema):  database, models, repository, importer, client, api
Legacy single-file SQLite signature store: SignatureLibrary (CLI --db)
"""
from .db import SignatureLibrary

__all__ = ["SignatureLibrary"]
