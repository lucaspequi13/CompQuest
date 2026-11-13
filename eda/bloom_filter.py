"""Bloom filter para deduplicar infrações (mesmo carro, mesma infração, vista por 2 câmeras).

Sem falso negativo; falso positivo controlado pelo tamanho do vetor de bits.
m = -(n ln p) / (ln 2)^2 bits, k = (m/n) ln 2 funções de hash.
"""
import hashlib
import math


class BloomFilter:
    def __init__(self, expected_items: int = 100_000, fp_rate: float = 0.01):
        self.m = max(8, int(-expected_items * math.log(fp_rate) / (math.log(2) ** 2)))
        self.k = max(1, round(self.m / expected_items * math.log(2)))
        self.bits = bytearray((self.m + 7) // 8)

    def _hashes(self, key: str):
        # double hashing: h_i = h1 + i*h2
        d = hashlib.sha256(key.encode()).digest()
        h1 = int.from_bytes(d[:8], "big")
        h2 = int.from_bytes(d[8:16], "big") | 1
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, key: str) -> None:
        for h in self._hashes(key):
            self.bits[h >> 3] |= 1 << (h & 7)

    def __contains__(self, key: str) -> bool:
        return all(self.bits[h >> 3] & (1 << (h & 7)) for h in self._hashes(key))

    def check_and_add(self, key: str) -> bool:
        """Retorna True se (provavelmente) já existia."""
        seen = key in self
        self.add(key)
        return seen


def violation_key(plate: str, violation_type: str, minute_bucket: int) -> str:
    """Mesma placa + mesmo tipo dentro da mesma janela de minutos = duplicata."""
    return f"{plate}|{violation_type}|{minute_bucket}"
