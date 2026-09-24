"""Deterministic public-domain/PSF-licensed definition corpus for Gate 2."""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable
import torch
from torch import Tensor

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\s]", re.UNICODE)


@dataclass(frozen=True)
class DefinitionRecord:
    record_id: str
    headword: str
    definition: str
    source_path: str
    qualified_name: str
    source_license: str = "Python Software Foundation License Version 2"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _clean(text: str, limit: int = 800) -> str:
    value = " ".join(str(text).split())
    return value[:limit].strip()


def scan_python_stdlib(root: Path, *, limit: int | None = 5000,
                       definition_limit: int = 800) -> tuple[DefinitionRecord, ...]:
    """Collect module, class, and function docstrings in stable source order.

    ``limit=None`` returns every docstring found. ``definition_limit`` is the
    character clip applied to each docstring (800 for the Gate 2 fixture).
    """
    candidates: list[DefinitionRecord] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if any(part in {"test", "tests", "site-packages", "dist-packages", "__pycache__"} for part in path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeError):
            continue
        module_name = relative[:-3].replace("/", ".")
        nodes: list[tuple[str, str, str]] = []
        module_doc = ast.get_docstring(tree, clean=True)
        if module_doc:
            nodes.append((module_name.rsplit(".", 1)[-1], module_name, module_doc))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node, clean=True)
            if not doc:
                continue
            qualified = f"{module_name}.{node.name}"
            nodes.append((node.name, qualified, doc))
        for headword, qualified, doc in sorted(nodes, key=lambda item: item[1]):
            definition = _clean(doc, definition_limit)
            if len(definition) < 24:
                continue
            identity = (headword.casefold(), definition.casefold())
            if identity in seen:
                continue
            seen.add(identity)
            digest = hashlib.sha256(f"{relative}\x1f{qualified}\x1f{definition}".encode()).hexdigest()[:24]
            candidates.append(DefinitionRecord(
                record_id=f"py_{digest}",
                headword=headword,
                definition=definition,
                source_path=relative,
                qualified_name=qualified,
            ))
    candidates.sort(key=lambda item: hashlib.sha256(item.record_id.encode()).hexdigest())
    if limit is None:
        return tuple(candidates)
    if len(candidates) < limit:
        raise RuntimeError(f"stdlib scan produced {len(candidates)} definitions; {limit} required")
    return tuple(candidates[:limit])


def write_definition_fixture(path: Path, records: Iterable[DefinitionRecord]) -> dict[str, object]:
    items = tuple(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in items:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "records": len(items),
        "source": "CPython standard-library source docstrings",
        "license": "Python Software Foundation License Version 2",
        "sha256": digest,
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def read_definition_fixture(path: Path) -> tuple[DefinitionRecord, ...]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.append(DefinitionRecord(**json.loads(line)))
    return tuple(result)


def fixture_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WordTokenizer:
    SPECIAL = ("<pad>", "<bos>", "<eos>", "<unk>", "<sep>")

    def __init__(self, tokens: tuple[str, ...]) -> None:
        if tokens[:len(self.SPECIAL)] != self.SPECIAL:
            raise ValueError("tokenizer special-token prefix differs from contract")
        self.tokens = tokens
        self.token_to_id = {value: index for index, value in enumerate(tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int: return 0
    @property
    def bos_id(self) -> int: return 1
    @property
    def eos_id(self) -> int: return 2
    @property
    def unk_id(self) -> int: return 3
    @property
    def sep_id(self) -> int: return 4

    @classmethod
    def build(cls, records: Iterable[DefinitionRecord], vocabulary_size: int) -> "WordTokenizer":
        if vocabulary_size <= len(cls.SPECIAL):
            raise ValueError("vocabulary is too small")
        counts: Counter[str] = Counter()
        for record in records:
            counts.update(cls.tokenize(record.headword))
            counts.update(cls.tokenize(record.definition))
        ordered = sorted(counts, key=lambda token: (-counts[token], token))
        return cls(tuple(cls.SPECIAL) + tuple(ordered[:vocabulary_size-len(cls.SPECIAL)]))

    @staticmethod
    def tokenize(text: str) -> tuple[str, ...]:
        return tuple(value.casefold() for value in _TOKEN.findall(text))

    def encode_text(self, text: str) -> tuple[int, ...]:
        return tuple(self.token_to_id.get(token, self.unk_id) for token in self.tokenize(text))

    def encode_record(self, record: DefinitionRecord, maximum_tokens: int = 64) -> tuple[int, ...]:
        headword = self.encode_text(record.headword)
        definition = self.encode_text(record.definition)
        values = (self.bos_id, *headword, self.sep_id, *definition, self.eos_id)
        if len(values) > maximum_tokens:
            values = (*values[:maximum_tokens-1], self.eos_id)
        return tuple(values)

    def definition_key(self, record: DefinitionRecord) -> tuple[int, ...]:
        values = self.encode_text(record.headword)
        return values if values else (self.unk_id,)

    def decode(self, token_ids: Iterable[int]) -> str:
        return " ".join(self.tokens[int(value)] if 0 <= int(value) < len(self.tokens) else "<oov>" for value in token_ids)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.tokens, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tokens": self.tokens, "fingerprint": self.fingerprint()}, ensure_ascii=False, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "WordTokenizer":
        record = json.loads(path.read_text())
        tokenizer = cls(tuple(record["tokens"]))
        if tokenizer.fingerprint() != record["fingerprint"]:
            raise ValueError("tokenizer fingerprint mismatch")
        return tokenizer


@dataclass(frozen=True)
class PublicCorpus:
    tokenizer: WordTokenizer
    training_records: tuple[DefinitionRecord, ...]
    validation_records: tuple[DefinitionRecord, ...]
    training_stream: Tensor
    validation_stream: Tensor


def build_public_corpus(records: tuple[DefinitionRecord, ...], *, vocabulary_size: int = 4096,
                        maximum_record_tokens: int = 64) -> PublicCorpus:
    if len(records) != 5000:
        raise ValueError("Gate 2 requires exactly 5,000 definitions")
    ordered = sorted(records, key=lambda record: hashlib.sha256(record.record_id.encode()).digest())
    training = tuple(ordered[:4000])
    validation = tuple(ordered[4000:])
    tokenizer = WordTokenizer.build(training, vocabulary_size)
    def stream(items: tuple[DefinitionRecord, ...]) -> Tensor:
        values: list[int] = []
        for record in items:
            values.extend(tokenizer.encode_record(record, maximum_record_tokens))
        return torch.tensor(values, dtype=torch.long)
    return PublicCorpus(tokenizer, training, validation, stream(training), stream(validation))


def stream_windows(values: Tensor, *, sequence_length: int, stride: int) -> tuple[Tensor, Tensor]:
    if values.ndim != 1 or values.numel() <= sequence_length:
        raise ValueError("token stream must be one-dimensional and longer than the window")
    windows = values.unfold(0, sequence_length + 1, stride)
    return windows[:, :-1].contiguous(), windows[:, 1:].contiguous()


def top_stream_phrases(values: Tensor, *, per_order: int, blocked_tokens: set[int]) -> tuple[list[tuple[int, ...]], Counter[tuple[int, ...]]]:
    sequence = values.tolist()
    counts: Counter[tuple[int, ...]] = Counter()
    selected: list[tuple[int, ...]] = []
    for order in (2, 3):
        local: Counter[tuple[int, ...]] = Counter()
        for start in range(0, len(sequence)-order+1):
            key = tuple(sequence[start:start+order])
            if any(value in blocked_tokens for value in key):
                continue
            local[key] += 1
        keys = sorted(local, key=lambda key: (-local[key], key))[:per_order]
        if len(keys) != per_order:
            raise RuntimeError(f"only {len(keys)} order-{order} phrases available")
        selected.extend(keys)
        counts.update(local)
    return selected, counts
