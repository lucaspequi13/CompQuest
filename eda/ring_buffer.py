"""Fila circular de tamanho fixo para os últimos N frames.

Guarda o "antes" de uma infração: quando a regra dispara, copiamos o buffer
para montar o clipe de evidência. push O(1), sem realocar memória.
"""
from typing import Generic, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity deve ser > 0")
        self._data: List[Optional[T]] = [None] * capacity
        self._cap = capacity
        self._head = 0  # próxima posição de escrita
        self._size = 0

    def push(self, item: T) -> None:
        self._data[self._head] = item
        self._head = (self._head + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def __len__(self) -> int:
        return self._size

    def is_full(self) -> bool:
        return self._size == self._cap

    def __iter__(self) -> Iterator[T]:
        """Do mais antigo para o mais novo."""
        start = (self._head - self._size) % self._cap
        for i in range(self._size):
            yield self._data[(start + i) % self._cap]  # type: ignore[misc]

    def snapshot(self) -> List[T]:
        return list(self)

    def latest(self) -> Optional[T]:
        if self._size == 0:
            return None
        return self._data[(self._head - 1) % self._cap]
