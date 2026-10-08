from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    start: int
    stop: int
    window_start: int
    window_stop: int
    context: int

    @property
    def primary(self) -> range:
        return range(self.start, self.stop)

    @property
    def inputs(self) -> range:
        return range(
            max(self.window_start, self.start - self.context), min(self.window_stop, self.stop + self.context)
        )

    def split(self) -> tuple[Chunk, Chunk]:
        if self.stop - self.start <= 1:
            raise ValueError("Cannot split a single primary frame")
        middle = (self.start + self.stop) // 2
        return (
            Chunk(self.start, middle, self.window_start, self.window_stop, self.context),
            Chunk(middle, self.stop, self.window_start, self.window_stop, self.context),
        )


def chunks(start: int, length: int = 250, primary: int = 25, context: int = 5):
    if length < 1 or primary < 1 or context < 0:
        raise ValueError("Invalid chunk dimensions")
    for offset in range(0, length, primary):
        yield Chunk(
            start + offset, min(start + length, start + offset + primary), start, start + length, context
        )
