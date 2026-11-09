from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from .units import (  # noqa: F401
        AngleUnit,  # noqa: F401, RUF100
        AngularMomentumUnit,  # noqa: F401, RUF100
        AngularVelocityUnit,  # noqa: F401, RUF100
        CurrentUnit,  # noqa: F401, RUF100
        DimensionlessUnit,  # noqa: F401, RUF100
        DistanceUnit,  # noqa: F401, RUF100
        EnergyUnit,  # noqa: F401, RUF100
        ForceUnit,  # noqa: F401, RUF100
        LinearAccelerationUnit,  # noqa: F401, RUF100
        LinearMomentumUnit,  # noqa: F401, RUF100
        LinearVelocityUnit,  # noqa: F401, RUF100
        MassUnit,  # noqa: F401, RUF100
        TimeUnit,  # noqa: F401, RUF100
        TorqueUnit,  # noqa: F401, RUF100
        Unit,  # noqa: F401, RUF100
        VoltageUnit,  # noqa: F401, RUF100
    )


A = TypeVar("A", bound="Unit")
B = TypeVar("B", bound="Unit")


@runtime_checkable
class Measure[U: "Unit"](Protocol):
    EQUIVALENCE_THRESHOLD: float = 1e-12

    def magnitude(self) -> float: ...
    def base_unit_magnitude(self) -> float: ...
    def unit(self) -> U: ...

    def in_unit(self, unit: U) -> float:
        if self.unit() == unit:
            return self.magnitude()
        else:
            return unit.from_base_units(self.base_unit_magnitude())

    def base_unit(self) -> U:
        return self.unit().get_base_unit()

    def abs(self, unit: U) -> float:
        return abs(self.in_unit(unit))

    def copy_sign(self, other: Measure[U], unit: U) -> float:
        import math
        return math.copysign(self.in_unit(unit), other.in_unit(unit))

    def unary_minus(self) -> Measure[U]: ...
    def plus(self, other: Measure[Any]) -> Measure[U]: ...
    def minus(self, other: Measure[Any]) -> Measure[U]: ...
    def times(self, multiplier: float | Measure[Any]) -> Measure[Any]: ...
    def div(self, divisor: float | Measure[Any]) -> Measure[Any]: ...


class ImmutableMeasure[U: "Unit"]:
    def __init__(
        self, magnitude: float, base_unit_magnitude: float, unit: U
    ) -> None:
        self._magnitude = magnitude
        self._base_unit_magnitude = base_unit_magnitude
        self._unit = unit

    def magnitude(self) -> float:
        return self._magnitude

    def base_unit_magnitude(self) -> float:
        return self._base_unit_magnitude

    def unit(self) -> U:
        return self._unit

    @staticmethod
    def of_base_units(
        base_unit_magnitude: float,
        unit: U
    ) -> ImmutableMeasure[U]:
        return ImmutableMeasure(
            unit.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            unit,
        )

    @staticmethod
    def of_relative_units(
        relative_magnitude: float,
        unit: U
    ) -> ImmutableMeasure[U]:
        return ImmutableMeasure(
            relative_magnitude, unit.to_base_units(relative_magnitude), unit
        )


class Dimensionless(ImmutableMeasure["Unit"]):
    pass


class Distance(ImmutableMeasure["Unit"]):
    pass


class Time(ImmutableMeasure["Unit"]):
    pass


class Mass(ImmutableMeasure["Unit"]):
    pass


class Angle(ImmutableMeasure["Unit"]):
    @staticmethod
    def of_relative_units(
        relative_magnitude: float,
        unit: "Unit"  # noqa: UP037
    ) -> Angle:
        return Angle(
            relative_magnitude,
            unit.to_base_units(relative_magnitude),
            unit
        )

    @staticmethod
    def of_base_units(
        base_unit_magnitude: float,
        unit: "Unit"  # noqa: UP037
    ) -> Angle:
        return Angle(
            unit.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            unit
        )


class Mult[A, B](ImmutableMeasure["Unit"]):
    pass


class Per[A, B](ImmutableMeasure["Unit"]):
    pass


class Voltage(ImmutableMeasure["Unit"]):
    pass


class Current(ImmutableMeasure["Unit"]):
    pass


class Energy(ImmutableMeasure["Unit"]):
    pass


class Temperature(ImmutableMeasure["Unit"]):
    pass


class Velocity[U: "Unit"](Per[U, "TimeUnit"]):
    pass


class LinearVelocity(Per["DistanceUnit", "TimeUnit"]):
    pass


class AngularVelocity(Per["AngleUnit", "TimeUnit"]):
    pass


class Frequency(Per["DimensionlessUnit", "TimeUnit"]):
    pass


class LinearAcceleration(Per["LinearVelocityUnit", "TimeUnit"]):
    pass


class AngularAcceleration(Per["AngularVelocityUnit", "TimeUnit"]):
    pass


class Force(Per["MassUnit", "LinearAccelerationUnit"]):
    pass


class Torque(Per["DistanceUnit", "ForceUnit"]):
    pass


class LinearMomentum(Per["MassUnit", "LinearVelocityUnit"]):
    pass


class AngularMomentum(Per["LinearMomentumUnit", "DistanceUnit"]):
    pass


class MomentOfInertia(Per["AngularMomentumUnit", "AngularVelocityUnit"]):
    pass


class Power(Per[
    "EnergyUnit|VoltageUnit|CurrentUnit|AngularVelocityUnit|TorqueUnit",
    "TimeUnit|CurrentUnit|VoltageUnit|TorqueUnit|AngularVelocityUnit"
]):
    pass


class Resistance(Per["VoltageUnit", "CurrentUnit"]):
    pass
