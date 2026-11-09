from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any


class ReadOnlyPrimitiveLongSet:
    """A read-only set of unique primitive `long` values."""

    def __init__(self, *values: int) -> None:  # noqa: S3776
        unique_values = [0] * len(values)
        num_unique_values = 0
        seen_zero = False

        for value in values:
            if value == 0 and not seen_zero:
                seen_zero = True
            else:
                exists = False
                for j in range(num_unique_values):
                    if unique_values[j] == value:
                        exists = True
                        break
                if exists:
                    continue
            unique_values[num_unique_values] = value
            num_unique_values += 1

        if num_unique_values == len(values):
            self._m_values = unique_values
        else:
            self._m_values = unique_values[:num_unique_values]

    def contains(self, value: int) -> bool:
        return value in self._m_values

    def size(self) -> int:
        return len(self._m_values)

    def is_empty(self) -> bool:
        return len(self._m_values) == 0

    def stream(self) -> list[int]:
        return list(self._m_values)

    def to_array(self) -> list[int]:
        return list(self._m_values)

    def __iter__(self) -> Iterator[int]:
        return iter(self._m_values)


class LongToObjectHashMap[V: Any | None]:
    """A variant on a HashMap that uses primitive long ints for map keys."""

    INITIAL_CAPACITY: int = 8
    LOAD_FACTOR: float = 0.75

    def __init__(self) -> None:
        self._m_size = 0
        self._m_capacity = self.INITIAL_CAPACITY
        self._m_keys: list[int] = [0] * self._m_capacity
        self._m_uses: list[bool] = [False] * self._m_capacity
        self._m_values: list[V | None] = [None] * self._m_capacity

    def _hash(self, key: int) -> int:
        I16_MAX = 0xFFFFFFFFFFFFFFFF
        return 31 + (
            key ^ (key >> 15) ^ (key >> 31) ^ ((key << 31) & I16_MAX)
        )

    def _mask(self) -> int:
        return self._m_capacity - 1

    def _bucket(self, key: int) -> int:
        return int(self._hash(key) & self._mask())

    def _safe_increment(self, bucket: int) -> int:
        return (bucket + 1) & self._mask()

    def _max_size(self) -> int:
        return int(self._m_capacity * self.LOAD_FACTOR)

    def put(self, key: int, value: V) -> V | None:
        bucket = self._bucket(key)
        while self._m_uses[bucket]:
            if self._m_keys[bucket] == key:
                old_value = self._m_values[bucket]
                self._m_values[bucket] = value
                return old_value
            bucket = self._safe_increment(bucket)

        self._m_uses[bucket] = True
        self._m_keys[bucket] = key
        self._m_values[bucket] = value
        self._m_size += 1

        if self._m_size > self._max_size():
            self._grow()
        return None

    def get(self, key: int) -> V | None:
        bucket = self._bucket(key)
        while self._m_uses[bucket]:
            if self._m_keys[bucket] == key:
                return self._m_values[bucket]
            bucket = self._safe_increment(bucket)
        return None

    def remove(self, key: int) -> V | None:
        bucket = self._bucket(key)
        while self._m_uses[bucket]:
            if self._m_keys[bucket] == key:
                self._m_size -= 1
                self._m_keys[bucket] = 0
                self._m_uses[bucket] = False
                old_value = self._m_values[bucket]
                self._m_values[bucket] = None
                return old_value
            bucket = self._safe_increment(bucket)
        return None

    def contains_key(self, key: int) -> bool:
        bucket = self._bucket(key)
        while self._m_uses[bucket]:
            if self._m_keys[bucket] == key:
                return True
            bucket = self._safe_increment(bucket)
        return False

    def clear(self) -> None:
        if self._m_size == 0:
            return
        self._m_size = 0
        self._m_uses = [False] * self._m_capacity
        self._m_keys = [0] * self._m_capacity
        self._m_values = [None] * self._m_capacity

    def size(self) -> int:
        return self._m_size

    def capacity(self) -> int:
        return self._m_capacity

    def is_empty(self) -> bool:
        return self._m_size == 0

    def key_set(self) -> ReadOnlyPrimitiveLongSet:
        keys = [0] * self._m_size
        i = 0
        for bucket in range(self._m_capacity):
            if self._m_uses[bucket]:
                keys[i] = self._m_keys[bucket]
                i += 1
        return ReadOnlyPrimitiveLongSet(*keys)

    def values(self) -> list[V | None]:
        vals: list[V | None] = []
        for bucket in range(self._m_capacity):
            if self._m_uses[bucket] and self._m_values[bucket] is not None:
                vals.append(self._m_values[bucket])
        return list(vals)

    def for_each(
        self, function: Callable[[int, V | None], None]
    ) -> None:
        for bucket in range(self._m_capacity):
            if self._m_uses[bucket] and self._m_values[bucket] is not None:
                function(self._m_keys[bucket], self._m_values[bucket])

    def _grow(self) -> None:
        current_size = self._m_size
        old_capacity = self._m_capacity
        if old_capacity * self.LOAD_FACTOR >= current_size:
            return

        new_capacity = old_capacity * 2
        new_mask = new_capacity - 1

        old_uses = self._m_uses
        old_keys = self._m_keys
        old_values = self._m_values

        new_uses = [False] * new_capacity
        new_keys = [0] * new_capacity
        new_values: list[V | None] = [None] * new_capacity

        for old_bucket in range(old_capacity):
            if not old_uses[old_bucket]:
                continue
            key = old_keys[old_bucket]
            value = old_values[old_bucket]

            new_bucket = int(self._hash(key) & new_mask)
            while new_uses[new_bucket]:
                new_bucket = (new_bucket + 1) & new_mask

            new_uses[new_bucket] = True
            new_keys[new_bucket] = key
            new_values[new_bucket] = value

        self._m_capacity = new_capacity
        self._m_uses = new_uses
        self._m_keys = new_keys
        self._m_values = new_values
