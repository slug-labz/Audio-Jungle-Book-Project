"""Loading, kept deliberately boring.

Features come from .npy / .npz / .csv.  Labels and groups come from a metadata
table (csv / tsv / json) or from columns of the same csv as the features.
Rows are matched by position, which is what every embedding-extraction script
in this field produces, and the loader refuses to guess if the lengths differ.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

__all__ = ["load_features", "load_table", "resolve"]


def load_features(path: str, key: str | None = None, layer: int | None = None) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        X = np.load(path, mmap_mode="r")
    elif ext == ".npz":
        z = np.load(path)
        names = list(z.files)
        if key is None:
            if len(names) != 1:
                raise ValueError(f"{path} holds {names}; pass --key to choose one")
            key = names[0]
        X = z[key]
    elif ext in (".csv", ".tsv"):
        raise ValueError("for a csv, pass it as the features file AND give --feature-cols")
    else:
        raise ValueError(f"unsupported feature file: {ext}")
    X = np.asarray(X)
    if X.ndim == 3:
        if layer is None:
            raise ValueError(f"{path} is {X.shape} (layers, clips, dim); pass --layer")
        X = X[layer]
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got shape {X.shape}")
    return np.asarray(X, dtype=np.float64)


def load_table(path: str) -> list[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        obj = json.load(open(path))
        if isinstance(obj, dict):
            for k in ("meta", "rows", "records", "data"):
                if k in obj and isinstance(obj[k], list):
                    obj = obj[k]
                    break
        if not isinstance(obj, list):
            raise ValueError(f"{path}: expected a list of records")
        return obj
    delim = "\t" if ext == ".tsv" else ","
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter=delim))


def _col(rows: list[dict], name: str) -> np.ndarray:
    if name not in rows[0]:
        raise ValueError(f"column {name!r} not found. Available: {sorted(rows[0])[:25]}")
    return np.array([r[name] for r in rows], dtype=object).astype(str)


def _numeric_matrix(rows: list[dict], cols: list[str]) -> np.ndarray:
    out = np.empty((len(rows), len(cols)))
    for j, c in enumerate(cols):
        for i, r in enumerate(rows):
            v = r[c]
            out[i, j] = float(v) if v not in ("", None) else np.nan
    return out


def resolve(features: str, label: str, group: str, meta: str | None = None,
            feature_cols: str | None = None, key: str | None = None,
            layer: int | None = None):
    """Return (X, y, groups).  `feature_cols` is a comma list, a `prefix*`
    glob, or the word 'rest' meaning every column that is not label/group."""
    ext = os.path.splitext(features)[1].lower()
    if ext in (".csv", ".tsv") and feature_cols:
        rows = load_table(features)
        names = [c for c in rows[0] if c not in (label, group)]
        if feature_cols == "rest":
            cols = names
        elif feature_cols.endswith("*"):
            pre = feature_cols[:-1]
            cols = [c for c in names if c.startswith(pre)]
        else:
            cols = [c.strip() for c in feature_cols.split(",")]
        if not cols:
            raise ValueError(f"--feature-cols {feature_cols!r} matched nothing")
        X = _numeric_matrix(rows, cols)
        return X, _col(rows, label), _col(rows, group)

    X = load_features(features, key=key, layer=layer)
    if not meta:
        raise ValueError("--meta is required when features come from a .npy/.npz file")
    rows = load_table(meta)
    if len(rows) != len(X):
        raise ValueError(f"{len(X)} feature rows but {len(rows)} metadata rows -- "
                         f"they are matched by position and must be the same length")
    return X, _col(rows, label), _col(rows, group)
