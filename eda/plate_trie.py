"""Trie de placas + correção de OCR por distância de Levenshtein.

O OCR erra caracteres parecidos (0/O, 1/I, 8/B). A trie permite:
- busca por prefixo (placas que começam com "PQX") em O(L)
- busca aproximada: percorre a trie mantendo a linha da matriz de
  Levenshtein, podando ramos cujo custo mínimo já passou do limite.
Formato Mercosul: LLLNLNN (ex.: ABC1D23). Antigo: LLLNNNN.
"""
import re
from typing import Dict, List, Optional, Tuple

MERCOSUL = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")
ANTIGA = re.compile(r"^[A-Z]{3}[0-9]{4}$")

# Posições 0-2 são letras; 3, 5, 6 são dígitos; 4 é letra (Mercosul) ou dígito (antiga)
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A"}
TO_DIGIT = {v: k for k, v in TO_LETTER.items()} | {"D": "0", "Q": "0"}


def normalize(raw: str) -> str:
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if len(s) != 7:
        return s
    chars = list(s)
    for i in (0, 1, 2):
        chars[i] = TO_LETTER.get(chars[i], chars[i])
    for i in (3, 5, 6):
        chars[i] = TO_DIGIT.get(chars[i], chars[i])
    return "".join(chars)


def is_valid_plate(p: str) -> bool:
    return bool(MERCOSUL.match(p) or ANTIGA.match(p))


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class _Node:
    __slots__ = ("children", "terminal")

    def __init__(self):
        self.children: Dict[str, "_Node"] = {}
        self.terminal = False


class PlateTrie:
    def __init__(self):
        self.root = _Node()
        self.size = 0

    def insert(self, plate: str) -> None:
        node = self.root
        for ch in plate:
            node = node.children.setdefault(ch, _Node())
        if not node.terminal:
            node.terminal = True
            self.size += 1

    def __contains__(self, plate: str) -> bool:
        node = self._walk(plate)
        return bool(node and node.terminal)

    def _walk(self, prefix: str) -> Optional[_Node]:
        node = self.root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def with_prefix(self, prefix: str) -> List[str]:
        node = self._walk(prefix)
        out: List[str] = []
        if node:
            self._collect(node, prefix, out)
        return out

    def _collect(self, node: _Node, acc: str, out: List[str]) -> None:
        if node.terminal:
            out.append(acc)
        for ch, nxt in node.children.items():
            self._collect(nxt, acc + ch, out)

    def fuzzy(self, word: str, max_dist: int = 1) -> List[Tuple[str, int]]:
        results: List[Tuple[str, int]] = []
        first_row = list(range(len(word) + 1))
        for ch, child in self.root.children.items():
            self._fuzzy(child, ch, ch, word, first_row, max_dist, results)
        return sorted(results, key=lambda t: t[1])

    def _fuzzy(self, node, ch, acc, word, prev_row, max_dist, results):
        row = [prev_row[0] + 1]
        for j in range(1, len(word) + 1):
            row.append(min(row[j - 1] + 1, prev_row[j] + 1, prev_row[j - 1] + (word[j - 1] != ch)))
        if node.terminal and row[-1] <= max_dist:
            results.append((acc, row[-1]))
        if min(row) <= max_dist:  # poda
            for nch, nxt in node.children.items():
                self._fuzzy(nxt, nch, acc + nch, word, row, max_dist, results)
