"""Packed binary tensor representation and weight-provider interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .manifest import ParameterSpec, TensorShape


@dataclass(frozen=True, slots=True)
class BinaryTensor:
    """Row-major weights encoded as zero=-1 and one=+1."""

    shape: TensorShape
    packed_data: bytes

    def __post_init__(self) -> None:
        if len(self.packed_data) != self.shape.packed_byte_count:
            raise ValueError(
                f"expected {self.shape.packed_byte_count} packed bytes, "
                f"received {len(self.packed_data)}"
            )

    def is_positive(self, row: int, column: int) -> bool:
        if row < 0 or row >= self.shape.rows:
            raise IndexError("row is outside the tensor")
        if column < 0 or column >= self.shape.columns:
            raise IndexError("column is outside the tensor")
        bit_index = row * self.shape.columns + column
        return bool((self.packed_data[bit_index // 8] >> (bit_index % 8)) & 1)

    def sign(self, row: int, column: int) -> int:
        return 1 if self.is_positive(row, column) else -1


class WeightProvider(ABC):
    """Extension point for local, mapped, remote, or accelerator storage."""

    @abstractmethod
    def load(self, parameter: ParameterSpec) -> BinaryTensor:
        """Load and validate one packed parameter tensor."""


class DirectoryWeightProvider(WeightProvider):
    """Loads each tensor from ``<root>/<parameter-name>.bin`` on demand."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def load(self, parameter: ParameterSpec) -> BinaryTensor:
        if not _is_safe_parameter_name(parameter.name):
            raise ValueError(f"unsafe parameter name: {parameter.name!r}")
        path = self._root / f"{parameter.name}.bin"
        try:
            data = path.read_bytes()
        except OSError as error:
            raise OSError(f"unable to read binary tensor {path}") from error
        return BinaryTensor(parameter.shape, data)


def _is_safe_parameter_name(name: str) -> bool:
    return bool(name) and all(
        character.isascii()
        and (character.isalnum() or character in {".", "_", "-"})
        for character in name
    )
