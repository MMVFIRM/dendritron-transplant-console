"""Immutable one-word definition-sense vectors and sparse CPU lookup."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json, os
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from torch import Tensor


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class DefinitionVectorRecord:
    token_id: int
    vector: np.ndarray
    sense_id: str
    text: str
    frequency: int = 0


class DefinitionBankBuilder:
    @staticmethod
    def write(root: str | Path, records: Iterable[DefinitionVectorRecord], *,
              dtype: str = "float32", overwrite: bool = False,
              source_kind: str = "joint_definition") -> dict[str, object]:
        destination = Path(root)
        if destination.exists() and any(destination.iterdir()) and not overwrite:
            raise FileExistsError(destination)
        destination.mkdir(parents=True, exist_ok=True)
        items = list(records)
        if not items:
            raise ValueError("definition bank requires records")
        width = int(np.asarray(items[0].vector).shape[0])
        vectors_path = destination / "vectors.npy"
        vectors = np.lib.format.open_memmap(vectors_path, mode="w+", dtype=dtype,
                                             shape=(len(items), width))
        metadata_path = destination / "metadata.jsonl"
        with metadata_path.open("w", encoding="utf-8") as handle:
            for row, item in enumerate(items):
                vector = np.asarray(item.vector)
                if vector.shape != (width,) or item.token_id < 0:
                    raise ValueError("invalid definition record")
                vectors[row] = vector.astype(dtype)
                handle.write(json.dumps({"row": row, "token_id": int(item.token_id),
                                         "sense_id": item.sense_id, "text": item.text,
                                         "frequency": int(item.frequency)},
                                        ensure_ascii=False, separators=(",", ":")) + "\n")
        vectors.flush(); del vectors
        manifest = {
            "schema_version": 1, "kind": "definition_bank", "rows": len(items),
            "width": width, "dtype": str(np.dtype(dtype)), "source_kind": source_kind,
            "files": {
                "vectors": {"path": vectors_path.name, "bytes": vectors_path.stat().st_size,
                            "sha256": _sha(vectors_path)},
                "metadata": {"path": metadata_path.name, "bytes": metadata_path.stat().st_size,
                             "sha256": _sha(metadata_path)},
            },
        }
        temporary = destination / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
        os.replace(temporary, destination / "manifest.json")
        return manifest


@dataclass(frozen=True)
class DefinitionLookup:
    vectors: Tensor
    mask: Tensor
    row_indices: Tensor


class FrozenDefinitionBank:
    def __init__(self, root: str | Path, *, validate: bool = True) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root/"manifest.json").read_text())
        self.rows = int(self.manifest["rows"]); self.width = int(self.manifest["width"])
        files = self.manifest["files"]
        if validate:
            for record in files.values():
                path=self.root/record["path"]
                if path.stat().st_size != int(record["bytes"]) or _sha(path) != record["sha256"]:
                    raise ValueError(f"definition bank integrity mismatch: {path}")
        self.vectors = np.load(self.root/files["vectors"]["path"], mmap_mode="r")
        self.by_token: dict[int, list[int]] = {}
        self.metadata: list[dict[str, object]] = []
        with (self.root/files["metadata"]["path"]).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip(): continue
                record=json.loads(line); row=int(record["row"])
                self.metadata.append(record)
                self.by_token.setdefault(int(record["token_id"]), []).append(row)
        for rows in self.by_token.values():
            rows.sort(key=lambda row: (-int(self.metadata[row]["frequency"]), row))
        if len(self.metadata) != self.rows or self.vectors.shape != (self.rows,self.width):
            raise ValueError("definition bank shape mismatch")

    def resolve(self, input_ids: Tensor, *, max_senses: int = 4) -> DefinitionLookup:
        if input_ids.ndim != 2 or max_senses < 1:
            raise ValueError("definition lookup requires [B,T] and positive max_senses")
        batch,length=input_ids.shape
        values=np.zeros((batch,length,max_senses,self.width),dtype=np.float32)
        mask=np.zeros((batch,length,max_senses),dtype=np.bool_)
        rows=np.full((batch,length,max_senses),-1,dtype=np.int64)
        for b, sequence in enumerate(input_ids.detach().cpu().tolist()):
            for t, token in enumerate(sequence):
                for s,row in enumerate(self.by_token.get(int(token),())[:max_senses]):
                    values[b,t,s]=np.asarray(self.vectors[row],dtype=np.float32)
                    mask[b,t,s]=True; rows[b,t,s]=row
        device=input_ids.device
        return DefinitionLookup(torch.from_numpy(values).to(device),
                                torch.from_numpy(mask).to(device),
                                torch.from_numpy(rows).to(device))
