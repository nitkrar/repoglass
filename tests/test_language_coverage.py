"""Every shipped tags query must capture a definition and a call."""

from __future__ import annotations

import unittest

from repoglass.config import Settings
from repoglass.corpus import extract, languages
from repoglass.models import SourceFile

#: One chunkable sample per language.
SAMPLES: dict[tuple[str, str], tuple[str, str]] = {
    ("python", "a.py"): ("def unlock_vault(key):\n"
                         "    return crypto.open(key, rounds=100_000)\n", "unlock_vault"),
    ("javascript", "a.js"): ("function unlockVault(key) {\n"
                             "  return crypto.open(key, 100000);\n}\n", "unlockVault"),
    ("typescript", "a.ts"): ("export function unlockVault(key: string): Vault {\n"
                             "  return crypto.open(key, 100000);\n}\n", "unlockVault"),
    ("go", "a.go"): ("package vault\nfunc UnlockVault(key string) error {\n"
                     "\treturn crypto.Open(key, 100000)\n}\n", "UnlockVault"),
    ("rust", "a.rs"): ("pub fn unlock_vault(key: &str) -> Result<Vault> {\n"
                       "    crypto::open(key, 100_000)\n}\n", "unlock_vault"),
    ("c", "a.c"): ("int unlock_vault(const char *key) {\n"
                   "    return crypto_open(key, 100000);\n}\n", "unlock_vault"),
    ("cpp", "a.cpp"): ("int unlock_vault(const std::string &key) {\n"
                       "    return crypto_open(key, 100000);\n}\n", "unlock_vault"),
    ("php", "a.php"): ("<?php\nfunction unlock_vault($key) {\n"
                       "    return crypto_open($key, 100000);\n}\n", "unlock_vault"),
    ("swift", "a.swift"): ("func unlockVault(key: String) throws -> Vault {\n"
                           "    return try crypto.open(key, rounds: 100000)\n}\n", "unlockVault"),
    ("kotlin", "a.kt"): ("fun unlockVault(key: String): Vault {\n"
                         "    return crypto.open(key, 100000)\n}\n", "unlockVault"),
    ("java", "a.java"): ("class Vault {\n  public void unlockVault(String key) {\n"
                         "    crypto.open(key, 100000);\n  }\n}\n", "unlockVault"),
    ("ruby", "a.rb"): ("def unlock_vault(key)\n"
                       "  Crypto.open(key, 100_000)\nend\n", "unlock_vault"),
    ("csharp", "a.cs"): ("class Vault {\n  public void UnlockVault(string key) {\n"
                         "    Crypto.Open(key, 100000);\n  }\n}\n", "UnlockVault"),
    ("bash", "a.sh"): ("unlock_vault() {\n"
                       "  crypto_open \"$1\" 100000\n}\n", "unlock_vault"),
    ("lua", "a.lua"): ("function unlock_vault(key)\n"
                       "  return crypto.open(key, 100000)\nend\n", "unlock_vault"),
    ("elixir", "a.ex"): ("defmodule Vault do\n  def unlock_vault(key) do\n"
                         "    Crypto.open(key, 100_000)\n  end\nend\n", "unlock_vault"),
    ("scala", "a.scala"): ("class Vault {\n  def unlockVault(key: String): Result =\n"
                           "    crypto.open(key, 100000)\n}\n", "unlockVault"),
    ("haskell", "a.hs"): ("module Vault where\n\nunlockVault :: String -> IO Vault\n"
                          "unlockVault key = cryptoOpen key 100000\n", "unlockVault"),
    ("zig", "a.zig"): ("pub fn unlockVault(key: []const u8) !Vault {\n"
                       "    return crypto.open(key, 100000);\n}\n", "unlockVault"),
}


#: Expected call capture for each sample.
CALLS: dict[str, str] = {
    "bash": "crypto_open", "c": "crypto_open", "cpp": "crypto_open",
    "csharp": "Open", "elixir": "open", "go": "Open",
    "haskell": "cryptoOpen", "java": "open", "javascript": "open",
    "kotlin": "open", "lua": "open", "php": "crypto_open",
    "python": "open", "ruby": "open", "rust": "open",
    "scala": "open", "swift": "open", "typescript": "open",
    "zig": "open",
}


def _refs(lang: str, path: str, src: str) -> set[str]:
    f = SourceFile(path=path, mtime_ns=1, size=len(src), lang=lang)
    return {s.name for s in extract.extract(f, src, Settings()).symbols
            if s.tag == "ref"}


class LanguageCoverageTests(unittest.TestCase):
    def test_every_shipped_query_extracts_its_definition(self) -> None:
        bad = []
        for (lang, path), (src, want) in SAMPLES.items():
            f = SourceFile(path=path, mtime_ns=1, size=len(src), lang=lang)
            names = {s.name for s in extract.extract(f, src, Settings()).symbols
                     if s.tag == "def"}
            if want not in names:
                bad.append(f"{lang}: wanted {want!r}, got {sorted(names)}")
        self.assertEqual([], bad)

    def test_every_sample_language_is_indexable(self) -> None:
        """A sampled language must survive discovery."""
        from repoglass.corpus import discovery
        bad = [path for (lang, path) in SAMPLES
               if not discovery.is_indexable(path)]
        self.assertEqual([], bad)

    def test_the_sample_set_covers_every_shipped_query(self) -> None:
        """Adding a query requires a sample."""
        shipped = {p.name.removesuffix("-tags.scm")
                   for p in languages.QUERY_DIR.glob("*-tags.scm")}
        covered = {lang for lang, _ in SAMPLES} | {"markdown"}
        self.assertEqual(set(), shipped - covered)

    def test_every_sample_language_has_a_declared_call(self) -> None:
        """Adding a sample requires an expected call capture."""
        self.assertEqual({lang for lang, _ in SAMPLES}, set(CALLS))

    def test_every_shipped_query_finds_the_call_in_its_sample(self) -> None:
        """Compilation is not enough; each sample must yield the call."""
        bad = []
        for (lang, path), (src, _) in SAMPLES.items():
            found = _refs(lang, path, src)
            if CALLS[lang] not in found:
                bad.append(f"{lang}: wanted {CALLS[lang]!r}, got {sorted(found)}")
        self.assertEqual([], bad)


if __name__ == "__main__":
    unittest.main()
