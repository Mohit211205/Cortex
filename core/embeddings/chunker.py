"""
AST-based code chunker.
Splits source files into meaningful chunks (functions, classes, blocks)
rather than naive line-based splits.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import List


@dataclass
class CodeChunk:
    content: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str   # "function", "class", "block"


def chunk_python(source: str) -> List[CodeChunk]:
    chunks: List[CodeChunk] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _fallback_chunk(source, "python")

    lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno
            content = "\n".join(lines[start:end])
            chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
            if len(content.strip()) > 10:
                chunks.append(
                    CodeChunk(
                        content=content,
                        start_line=start + 1,
                        end_line=end,
                        language="python",
                        chunk_type=chunk_type,
                    )
                )

    if not chunks:
        return _fallback_chunk(source, "python")

    return chunks


def chunk_generic(source: str, language: str, window: int = 30, overlap: int = 5) -> List[CodeChunk]:
    """Sliding-window chunker for languages without AST support."""
    lines = source.splitlines()
    chunks: List[CodeChunk] = []
    i = 0
    while i < len(lines):
        end = min(i + window, len(lines))
        content = "\n".join(lines[i:end])
        if content.strip():
            chunks.append(
                CodeChunk(
                    content=content,
                    start_line=i + 1,
                    end_line=end,
                    language=language,
                    chunk_type="block",
                )
            )
        i += window - overlap

    return chunks


def _fallback_chunk(source: str, language: str) -> List[CodeChunk]:
    return chunk_generic(source, language)


LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".cpp": "cpp",
    ".c": "c",
    ".rs": "rust",
    ".rb": "ruby",
}


def chunk_file(content: str, filename: str) -> List[CodeChunk]:
    suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
    language = LANGUAGE_MAP.get(suffix, "unknown")

    if language == "python":
        return chunk_python(content)
    else:
        return chunk_generic(content, language)
