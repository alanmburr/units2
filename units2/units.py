from __future__ import annotations

import locale
import math
from collections.abc import Callable
from typing import Any, Final, Self, TypeVar, cast

from .collections import LongToObjectHashMap
from .measure import (
    Angle,
    AngularAcceleration,
    AngularMomentum,
    AngularVelocity,
    Current,
    Dimensionless,
    Distance,
    Energy,
    Force,
    Frequency,
    ImmutableMeasure,
    LinearAcceleration,
    LinearMomentum,
    LinearVelocity,
    Mass,
    Measure,
    MomentOfInertia,
    Mult,
    Per,
    Power,
    Resistance,
    Temperature,
    Time,
    Torque,
    Velocity,
    Voltage,
)

U = TypeVar("U", bound="Unit")
A = TypeVar("A", bound="Unit")
B = TypeVar("B", bound="Unit")
Out = TypeVar("Out", bound="Unit")


def _locale_invariant_lower(string: str):
    nowlocale = locale.getlocale(locale.LC_ALL)
    locale.setlocale(locale.LC_ALL, "C")
    try:
        return string.lower()
    finally:
        locale.setlocale(locale.LC_ALL, nowlocale)


class UnaryFunction:
    """A function that accepts a single float and returns a float."""

    IDENTITY: UnaryFunction

    def __init__(self, func: Callable[[float], float]) -> None:
        self._func = func

    def apply(self, input_val: float) -> float:
        return self._func(input_val)

    @staticmethod
    def _identity(x: float) -> float:
        return x

    def pipe_to(self, next_func: UnaryFunction) -> UnaryFunction:
        if next_func is None:
            raise ValueError(
                "The next operation in the chain must be provided"
            )
        return UnaryFunction(
            lambda x: next_func.apply(self.apply(x))
        )

    def mult(self, multiplier: UnaryFunction | float) -> UnaryFunction:
        if isinstance(multiplier, UnaryFunction):
            return UnaryFunction(
                lambda x: self.apply(x) * multiplier.apply(x)
            )
        else:
            return UnaryFunction(
                lambda x: self.apply(x) * multiplier
            )

    def div(self, divisor: UnaryFunction | float) -> UnaryFunction:
        if isinstance(divisor, UnaryFunction):
            return UnaryFunction(lambda x: (
                0.0 if self.apply(x) == 0
                else self.apply(x) / divisor.apply(x)
            ))
        else:
            return UnaryFunction(lambda x: self.apply(x) / divisor)

    def exp(self, exponent: UnaryFunction | float) -> UnaryFunction:
        if isinstance(exponent, UnaryFunction):
            return UnaryFunction(
                lambda x: math.pow(self.apply(x), exponent.apply(x))
            )
        else:
            return UnaryFunction(
                lambda x: math.pow(self.apply(x), exponent)
            )


UnaryFunction.IDENTITY = (
    UnaryFunction(UnaryFunction._identity)
)


class Unit:
    """Unit of measurement that defines a quantity."""

    def __init__(
        self,
        base_unit: Unit | None,
        to_base_converter: UnaryFunction | float,
        from_base_converter: UnaryFunction | float,
        name: str | None,
        symbol: str | None,
    ) -> None:
        self._base_unit = base_unit if base_unit is not None else self

        if isinstance(to_base_converter, (int, float)):
            eq = float(to_base_converter)
            self._to_base_converter = UnaryFunction(lambda x: x * eq)
            self._from_base_converter = UnaryFunction(lambda x: x / eq)
        else:
            self._to_base_converter = to_base_converter
            self._from_base_converter = (
                from_base_converter
                if isinstance(from_base_converter, UnaryFunction)
                else UnaryFunction(lambda x: x)
            )

        self._name: str = name or "(Unknown Unit)"
        self._symbol: str = symbol or "(Unknown Symbol)"
        self._zero = self.of(0.0)
        self._one = self.of(1.0)

    def of(self, magnitude: float) -> ImmutableMeasure[Unit] | Per[Unit, Unit]:
        raise NotImplementedError

    def of_base_units(
        self,
        base_unit_magnitude: float
    ) -> ImmutableMeasure[Unit] | Per[Unit, Unit]:
        raise NotImplementedError

    def zero(self) -> ImmutableMeasure[Unit]:
        return self._zero

    def one(self) -> ImmutableMeasure[Unit]:
        return self._one

    def per[U: Unit](self, other: U) -> Any | PerUnit[Self, U]:
        raise NotImplementedError

    def get_base_unit(self) -> Self:
        return cast(Self, self._base_unit)

    def is_base_unit(self) -> bool:
        return self == self._base_unit

    def from_base_units(self, value_in_base_units: float) -> float:
        return self._from_base_converter.apply(value_in_base_units)

    def to_base_units(self, value_in_native_units: float) -> float:
        return self._to_base_converter.apply(value_in_native_units)

    def get_converter_to_base(self) -> UnaryFunction:
        return self._to_base_converter

    def get_converter_from_base(self) -> UnaryFunction:
        return self._from_base_converter

    def equivalent(self, other: Unit) -> bool:
        if isinstance(other, type(self)):
            return False
        arbitrary = 16777.214
        return (
            abs(
                self._from_base_converter.apply(arbitrary)
                - other._from_base_converter.apply(arbitrary)
            )
            <= Measure.EQUIVALENCE_THRESHOLD
            and abs(
                self._to_base_converter.apply(arbitrary)
                - other._to_base_converter.apply(arbitrary)
            )
            <= Measure.EQUIVALENCE_THRESHOLD
        )

    def mult(self, other: Unit) -> Unit:
        raise NotImplementedError

    def __eq__(self, o: object) -> bool:
        return (
            self is o
            or isinstance(o, Unit)
            and self._name == o._name
            and self._symbol == o._symbol
            and self.equivalent(o)
        )

    def __hash__(self) -> int:
        return hash((
            id(self._to_base_converter),
            id(self._from_base_converter),
            self._name,
            self._symbol,
        ))

    def name(self) -> str:
        return self._name

    def symbol(self) -> str:
        return self._symbol

    def cast[U: Unit](self, typecast: type[U]) -> U:
        return cast(typecast, self)  # type: ignore

    def __str__(self) -> str:
        return self.name()


class CombinatoryUnitCache[A, B, Out]:
    """A helper class for creating and caching combined units."""

    def __init__(
        self,
        constructor: Callable[[A, B], Out]
    ) -> None:
        self._cache: LongToObjectHashMap[Out] = LongToObjectHashMap()
        self._constructor = constructor

    def combine(self, a: A, b: B) -> Out:
        key = (hash(a) << 32) | (hash(b) & 0xFFFFFFFF)
        existing = self._cache.get(key)
        if existing is not None:
            return existing

        new_unit = self._constructor(a, b)
        self._cache.put(key, new_unit)
        return new_unit


class UnitBuilder[A: Unit]:
    """Builder used for easily deriving new units."""

    def __init__(self, base: A) -> None:
        self._m_base = base
        self._m_from_base = UnaryFunction.IDENTITY
        self._m_to_base = UnaryFunction.IDENTITY
        self._m_name: str | None = None
        self._m_symbol: str | None = None

    def offset(self, offset_val: float) -> UnitBuilder[A]:
        self._m_to_base = UnaryFunction(lambda x: x + offset_val)
        self._m_from_base = UnaryFunction(lambda x: x - offset_val)
        return self

    class MappingBuilder:
        def __init__(
            self,
            outer: UnitBuilder,
            min_input: float,
            max_input: float,
        ) -> None:
            self._outer = outer
            self._min_input = min_input
            self._max_input = max_input

        def to_output_range(
            self, min_output: float, max_output: float
        ) -> UnitBuilder:
            def map_val(
                v: float,
                in_min: float,
                in_max: float,
                out_min: float,
                out_max: float
            ) -> float:
                return (
                    (v - in_min) * (out_max - out_min) / (in_max - in_min)
                    + out_min
                )

            self._outer._m_from_base = UnaryFunction(
                lambda x: map_val(
                    x, self._min_input, self._max_input, min_output, max_output
                )
            )
            self._outer._m_to_base = UnaryFunction(
                lambda y: map_val(
                    y, min_output, max_output, self._min_input, self._max_input
                )
            )
            return self._outer

    def mapping_input_range(
        self, min_base: float, max_base: float
    ) -> MappingBuilder:
        return UnitBuilder.MappingBuilder(self, min_base, max_base)

    def from_base(self, from_base: UnaryFunction) -> UnitBuilder[A]:
        self._m_from_base = from_base
        return self

    def to_base(self, to_base: UnaryFunction) -> UnitBuilder[A]:
        self._m_to_base = to_base
        return self

    def named(self, name: str) -> UnitBuilder[A]:
        self._m_name = name
        return self

    def symbol(self, symbol: str) -> UnitBuilder[A]:
        self._m_symbol = symbol
        return self

    def split_into(self, fraction: float) -> UnitBuilder[A]:
        if fraction == 0:
            raise ValueError("Fraction must be nonzero")
        return self.to_base(
            UnaryFunction(lambda x: x / fraction)
        ).from_base(
            UnaryFunction(lambda b: b * fraction)
        )

    def aggregate(self, aggregation: float) -> UnitBuilder[A]:
        if aggregation == 0:
            raise ValueError("Aggregation amount must be nonzero")
        return self.to_base(
            UnaryFunction(lambda x: x * aggregation)
        ).from_base(
            UnaryFunction(lambda b: b / aggregation)
        )

    def make(
        self,
        constructor: Callable[
            [Unit, UnaryFunction, UnaryFunction, str, str], Unit
        ] | None = None,
    ) -> Unit:
        if self._m_name is None or self._m_symbol is None:
            raise ValueError("Name and symbol must be set")

        base_unit: Unit = self._m_base.get_base_unit()
        to_base = self._m_to_base.pipe_to(self._m_base.get_converter_to_base())
        from_base = self._m_base.get_converter_from_base().pipe_to(
            self._m_from_base
        )

        if constructor is not None:
            return constructor(
                base_unit, to_base, from_base, self._m_name, self._m_symbol
            )
        else:
            return type(self._m_base)(
                base_unit, to_base, from_base, self._m_name, self._m_symbol
            )

    def make_and_cast[T: Unit](
        self,
        typecast: type[T],
        constructor: Callable[
            [Unit, UnaryFunction, UnaryFunction, str, str], Unit
        ] | None = None,
    ) -> T:
        return cast(typecast, self.make(constructor))  # type: ignore


class DimensionlessUnit(Unit):
    def __init__(
        self,
        base_unit: DimensionlessUnit | None,
        to_base: Any,
        from_base_or_name: Any,
        name: str | None = None,
        symbol: str | None = None,
    ) -> None:
        super().__init__(base_unit,
                         to_base, from_base_or_name,
                         name, symbol)

    def get_base_unit(self) -> DimensionlessUnit:
        return super().get_base_unit()

    def of(self, magnitude: float) -> Dimensionless:
        return Dimensionless(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(
        self,
        base_unit_magnitude: float
    ) -> Dimensionless:
        return Dimensionless(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )

    def zero(self) -> Dimensionless:
        return cast(Dimensionless, super().zero())

    def one(self) -> Dimensionless:
        return cast(Dimensionless, super().one())

    def convert_from(
        self,
        magnitude: float,
        other_unit: DimensionlessUnit
    ) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))

    def per[U: Unit](
        self, other: U
    ) -> FrequencyUnit | PerUnit[DimensionlessUnit, U]:
        if isinstance(other, TimeUnit):
            return cast(FrequencyUnit, FrequencyUnit.combine(self, other))
        return self.per_unit(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[DimensionlessUnit, U]:
        return PerUnit.combine(self, other)


class TimeUnit(Unit):
    def __init__(
        self,
        base_unit: TimeUnit | None,
        to_base: Any,
        from_base_or_name: Any,
        name: str | None = None,
        symbol: str | None = None,
    ) -> None:
        super().__init__(base_unit, to_base, from_base_or_name, name, symbol)

    def get_base_unit(self) -> TimeUnit:
        return super().get_base_unit()

    def per[U: Unit](
        self, other: U
    ) -> DimensionlessUnit | PerUnit[TimeUnit, U]:
        if isinstance(other, TimeUnit):
            val = Units.derive(Units.Value).to_base(
                self.get_converter_to_base().div(
                    other.get_converter_to_base())
            ).from_base(
                other.get_converter_from_base().div(
                    self.get_converter_from_base())
            ).named(self.name() + " per " + other.name()).symbol(
                self.symbol() + "/" + other.symbol()
            ).make()
            return cast(DimensionlessUnit, val)
        return self.per_unit(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[TimeUnit, U]:
        return PerUnit.combine(self, other)

    def convert_from(self, magnitude: float, other_unit: TimeUnit) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))

    def of(self, magnitude: float) -> Time:
        return Time(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(
        self,
        base_unit_magnitude: float
    ) -> Time:
        return Time(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )

    def zero(self) -> Time:
        return cast(Time, super().zero())

    def one(self) -> Time:
        return cast(Time, super().one())


class DistanceUnit(Unit):
    def __init__(
        self,
        base_unit: DistanceUnit | None,
        to_base: Any,
        from_base_or_name: Any,
        name: str | None = None,
        symbol: str | None = None,
    ) -> None:
        super().__init__(base_unit, to_base, from_base_or_name, name, symbol)

    def get_base_unit(self) -> DistanceUnit:
        return super().get_base_unit()

    def per[U: Unit](
        self, other: U
    ) -> LinearVelocityUnit[DistanceUnit] | PerUnit[DistanceUnit, U]:
        if isinstance(other, TimeUnit):
            return LinearVelocityUnit.combine(self, other)
        return self.per_unit(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[DistanceUnit, U]:
        return PerUnit.combine(self, other)

    def convert_from(
        self,
        magnitude: float,
        other_unit: DistanceUnit
    ) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))

    def of(self, magnitude: float) -> Distance:
        return Distance(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(
        self,
        base_unit_magnitude: float
    ) -> Distance:
        return Distance(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )

    def zero(self) -> Distance:
        return cast(Distance, super().zero())

    def one(self) -> Distance:
        return cast(Distance, super().one())

    def mult_as_torque(
        self,
        force: ForceUnit[
            MassUnit,
            LinearAccelerationUnit[LinearVelocityUnit[DistanceUnit]]
        ]
    ) -> TorqueUnit[
        DistanceUnit,
        ForceUnit[
            MassUnit,
            LinearAccelerationUnit[LinearVelocityUnit[DistanceUnit]]
        ]
    ]:
        return TorqueUnit.combine(self, force)


class MassUnit(Unit):
    def __init__(
        self,
        base_unit: MassUnit | None,
        to_base: Any,
        from_base_or_name: Any,
        name: str | None = None,
        symbol: str | None = None,
    ) -> None:
        super().__init__(base_unit, to_base, from_base_or_name, name, symbol)

    def get_base_unit(self) -> MassUnit:
        return super().get_base_unit()

    def of(self, magnitude: float) -> Mass:
        return Mass(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(
        self,
        base_unit_magnitude: float
    ) -> Mass:
        return Mass(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )

    def zero(self) -> Mass:
        return cast(Mass, super().zero())

    def one(self) -> Mass:
        return cast(Mass, super().one())

    def per[U: Unit](
        self, other: U
    ) -> VelocityUnit[MassUnit] | PerUnit[MassUnit, U]:
        if isinstance(other, TimeUnit):
            return VelocityUnit.combine(self, other)
        return self.per_unit(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[MassUnit, U]:
        return PerUnit.combine(self, other)

    def convert_from(self, magnitude: float, other_unit: MassUnit) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))

    def mult(self, other: Unit) -> Unit:
        if isinstance(other, LinearVelocityUnit):
            return LinearMomentumUnit.combine(self, other)
        elif isinstance(other, LinearAccelerationUnit):
            return ForceUnit.combine(self, other)
        raise TypeError("Invalid argument for mult")


class AngleUnit(Unit):
    def __init__(
        self,
        base_unit: AngleUnit | None,
        to_base: Any,
        from_base_or_name: Any,
        name: str | None = None,
        symbol: str | None = None,
    ) -> None:
        super().__init__(base_unit, to_base, from_base_or_name, name, symbol)

    def get_base_unit(self) -> AngleUnit:
        return super().get_base_unit()

    def of(self, magnitude: float) -> Angle:
        return Angle.of_relative_units(magnitude, self)

    def of_base_units(
        self,
        base_unit_magnitude: float
    ) -> Angle:
        return Angle.of_base_units(base_unit_magnitude, self)

    def per[U: Unit](
        self, other: U
    ) -> AngularVelocityUnit[AngleUnit] | PerUnit[AngleUnit, U]:
        if isinstance(other, TimeUnit):
            return AngularVelocityUnit.combine(self, other)
        return self.per_unit(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[AngleUnit, U]:
        return PerUnit.combine(self, other)

    def convert_from(self, magnitude: float, other_unit: AngleUnit) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))

    def zero(self) -> Angle:
        return cast(Angle, super().zero())

    def one(self) -> Angle:
        return cast(Angle, super().one())


class VoltageUnit(Unit):
    def __init__(
        self,
        base_unit: VoltageUnit | None,
        to_base: Any,
        from_base_or_name: Any,
        name: str | None = None,
        symbol: str | None = None,
    ) -> None:
        super().__init__(base_unit, to_base, from_base_or_name, name, symbol)

    def get_base_unit(self) -> VoltageUnit:
        return super().get_base_unit()

    def mult(  # type: ignore[override]
        self, current: CurrentUnit, name: str, symbol: str  # noqa: S2638
    ) -> PowerUnit:
        return cast(PowerUnit, Units.derive(
            PowerUnit.combine(self, current)
        ).named(name).symbol(symbol).make())

    def of(self, magnitude: float) -> Voltage:
        return Voltage(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(
        self,
        base_unit_magnitude: float
    ) -> Voltage:
        return Voltage(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )

    def zero(self) -> Voltage:
        return cast(Voltage, super().zero())

    def one(self) -> Voltage:
        return cast(Voltage, super().one())

    def per(
        self, other: Unit
    ) -> (
        ResistanceUnit
        | VelocityUnit[VoltageUnit]
        | PerUnit[VoltageUnit, Unit]
    ):
        if isinstance(other, CurrentUnit):
            return ResistanceUnit.combine(self, other)
        elif isinstance(other, TimeUnit):
            return VelocityUnit.combine(self, other)
        return self.per_unit(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[VoltageUnit, U]:
        return PerUnit.combine(self, other)

    def convert_from(self, magnitude: float, other_unit: VoltageUnit) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))


class CurrentUnit(Unit):
    def __init__(
        self,
        base_unit: CurrentUnit | None,
        to_base: Any,
        from_base_or_name: Any,
        name: str | None = None,
        symbol: str | None = None,
    ) -> None:
        super().__init__(base_unit, to_base, from_base_or_name, name, symbol)

    def get_base_unit(self) -> CurrentUnit:
        return super().get_base_unit()

    def mult(  # type: ignore[override]
        self,
        voltage: VoltageUnit,  # noqa: S2638
        name: str,  # noqa: S2638
        symbol: str  # noqa: S2638
    ) -> PowerUnit[EnergyUnit, TimeUnit]:
        if name is None or symbol is None:
            raise ValueError(
                "This implementation of mult "
                "may not be the one you're looking for..."
            )
        return cast(PowerUnit, Units.derive(
            PowerUnit.combine(voltage, self)
        ).named(name).symbol(symbol).make())

    def of(self, magnitude: float) -> Current:
        return Current(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(
        self,
        base_unit_magnitude: float
    ) -> Current:
        return Current(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )

    def zero(self) -> Current:
        return cast(Current, super().zero())

    def one(self) -> Current:
        return cast(Current, super().one())

    def per(
        self, other: Unit
    ) -> VelocityUnit[CurrentUnit] | PerUnit[CurrentUnit, Unit]:
        if isinstance(other, TimeUnit):
            return VelocityUnit.combine(self, other)
        return self.per_unit(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[CurrentUnit, U]:
        return PerUnit.combine(self, other)

    def convert_from(self, magnitude: float, other_unit: CurrentUnit) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))


class EnergyUnit(Unit):
    def __init__(
        self,
        base_unit: EnergyUnit | None,
        to_base: Any,
        from_base_or_name: Any,
        name: str | None = None,
        symbol: str | None = None,
    ) -> None:
        super().__init__(base_unit, to_base, from_base_or_name, name, symbol)

    def get_base_unit(self) -> EnergyUnit:
        return super().get_base_unit()

    def per[U: Unit](self, other: U) -> PowerUnit | PerUnit[EnergyUnit, U]:
        if isinstance(other, TimeUnit):
            return PowerUnit.combine(self, other)
        return self.per_unit(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[EnergyUnit, U]:
        return PerUnit.combine(self, other)

    def convert_from(self, magnitude: float, other_unit: EnergyUnit) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))

    def of(self, magnitude: float) -> Energy:
        return Energy(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(
        self,
        base_unit_magnitude: float
    ) -> Energy:
        return Energy(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )

    def zero(self) -> Energy:
        return cast(Energy, super().zero())

    def one(self) -> Energy:
        return cast(Energy, super().one())


class TemperatureUnit(Unit):
    def __init__(
        self,
        base_unit: TemperatureUnit | None,
        to_base: Any,
        from_base_or_name: Any,
        name: str | None = None,
        symbol: str | None = None,
    ) -> None:
        super().__init__(base_unit, to_base, from_base_or_name, name, symbol)

    def get_base_unit(self) -> TemperatureUnit:
        return super().get_base_unit()

    def of(self, magnitude: float) -> Temperature:
        return Temperature(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(
        self,
        base_unit_magnitude: float
    ) -> Temperature:
        return Temperature(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )

    def zero(self) -> Temperature:
        return cast(Temperature, super().zero())

    def one(self) -> Temperature:
        return cast(Temperature, super().one())

    def per[U: Unit](
        self, other: U
    ) -> VelocityUnit[TemperatureUnit] | PerUnit[TemperatureUnit, U]:
        if isinstance(other, TimeUnit):
            return VelocityUnit.combine(self, other)
        return self.per_unit(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[TemperatureUnit, U]:
        return PerUnit.combine(self, other)

    def convert_from(
        self, magnitude: float, other_unit: TemperatureUnit
    ) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))


class PerUnit[A: Unit, B: Unit](Unit):
    def __init__(
        self,
        base_unit: PerUnit[A, B] | None,
        numerator: A,
        denominator: B,
    ) -> None:
        self._m_numerator = numerator
        self._m_denominator = denominator
        super().__init__(
            base_unit,
            numerator.to_base_units(1.0) / denominator.to_base_units(1.0),
            numerator.from_base_units(1.0) * denominator.to_base_units(1.0),
            numerator.name() + " per " + denominator.name(),
            numerator.symbol() + "/" + denominator.symbol(),
        )
        self._base_unit = base_unit if base_unit is not None else self

    def get_base_unit(self) -> PerUnit[A, B]:
        return super().get_base_unit()

    def numerator(self) -> A:
        return self._m_numerator

    def denominator(self) -> B:
        return self._m_denominator

    def reciprocal(self) -> VelocityUnit[B] | PerUnit[B, A]:
        if isinstance(self._m_numerator, TimeUnit):
            return VelocityUnit.combine(
                self._m_denominator, self._m_numerator
            )
        return PerUnit.combine(self._m_denominator, self._m_numerator)

    def mult(self, denom: B) -> Unit:  # type: ignore[override]
        if denom.equivalent(self.denominator()):
            return self.numerator()

        return (
            Units.derive(self.numerator())
            .to_base(denom.get_converter_to_base().div(
                self.denominator().get_converter_to_base()
            ))
            .from_base(denom.get_converter_from_base().div(
                self.denominator().get_converter_from_base()
            ))
            .named(f"{self.name()} {denom.name()}")
            .symbol(f"{self.symbol()}-{denom.symbol()}")
            .make()
        )

    def of(self, magnitude: float) -> Per[A, B]:
        return self.of_native(magnitude)

    def of_base_units(
        self,
        base_unit_magnitude: float
    ) -> Per[A, B]:
        return self.of_native_base_units(base_unit_magnitude)

    def of_native(self, magnitude: float) -> Per[A, B]:
        return Per(magnitude, self.to_base_units(magnitude), self)

    def of_native_base_units(
        self, base_unit_magnitude: float
    ) -> Per[A, B]:
        return Per(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )

    @staticmethod
    def combine(numerator: A, denominator: B) -> PerUnit[A, B]:
        return PerUnitCache.combine(numerator, denominator)

    def convert_from(
        self, magnitude: float, other_unit: Unit
    ) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))


PerUnitCache: CombinatoryUnitCache[Any, Any, PerUnit[Any, Any]] = (
    CombinatoryUnitCache(lambda n, d: PerUnit(None, n, d))
)


class MultUnit[A: Unit, B: Unit](Unit):
    def __init__(
        self,
        base_unit: MultUnit[A, B] | None,
        a: A,
        b: B,
    ) -> None:
        self._m_unitA = a
        self._m_unitB = b
        super().__init__(
            base_unit,
            a.to_base_units(1.0) * b.to_base_units(1.0),
            a.from_base_units(1.0) / b.from_base_units(1.0),
            a.name() + "-" + b.name(),
            a.symbol() + "*" + b.symbol(),
        )

    def get_base_unit(self) -> MultUnit[A, B]:
        return super().get_base_unit()

    def unit_a(self) -> A:
        return self._m_unitA

    def unit_b(self) -> B:
        return self._m_unitB

    @staticmethod
    def combine(a: A, b: B) -> MultUnit[A, B]:
        return MultUnitCache.combine(a, b)

    def of(self, magnitude: float) -> Mult[A, B]:
        return self.of_native(magnitude)

    def of_base_units(
        self,
        base_unit_magnitude: float
    ) -> Mult[A, B]:
        return self.of_native_base_units(base_unit_magnitude)

    def of_native(self, magnitude: float) -> Mult[A, B]:
        return Mult(magnitude, self.to_base_units(magnitude), self)

    def of_native_base_units(
        self, base_unit_magnitude: float
    ) -> Mult[A, B]:
        return Mult(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )


MultUnitCache: CombinatoryUnitCache[Any, Any, MultUnit[Any, Any]] = (
    CombinatoryUnitCache(lambda a, b: MultUnit(None, a, b))
)


class FrequencyUnit[U: DimensionlessUnit](PerUnit[U, TimeUnit]):
    def __init__(
        self, numerator: U, denominator: TimeUnit
    ) -> None:
        base = (
            None
            if numerator.is_base_unit() and denominator.is_base_unit()
            else FrequencyUnit.combine(
                numerator.get_base_unit(), denominator.get_base_unit()
            )
        )
        super().__init__(base, numerator, denominator)

    @staticmethod
    def combine(
        numerator: DimensionlessUnit,
        denominator: TimeUnit
    ) -> FrequencyUnit:
        if not isinstance(numerator, DimensionlessUnit):
            raise TypeError(
                "Numerator must be dimensionless in a FrequencyUnit"
            )
        if not isinstance(denominator, TimeUnit):
            raise TypeError(
                "Denominator must be a time unit in a FrequencyUnit"
            )
        return FrequencyUnitCache.combine(numerator, denominator)

    def get_base_unit(self) -> FrequencyUnit:
        return cast(FrequencyUnit, super().get_base_unit())

    def of(self, magnitude: float) -> Frequency:  # type: ignore[override]
        return Frequency(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(  # type: ignore[override]
        self,
        base_unit_magnitude: float
    ) -> Frequency:
        return Frequency(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )

    def per(self, other: Unit) -> VelocityUnit[FrequencyUnit]:
        if isinstance(other, TimeUnit):
            return VelocityUnit.combine(self, other)
        raise NotImplementedError


FrequencyUnitCache: CombinatoryUnitCache[
    DimensionlessUnit, TimeUnit, FrequencyUnit
] = CombinatoryUnitCache(FrequencyUnit)


class LinearVelocityUnit[U: DistanceUnit](PerUnit[U, TimeUnit]):
    def __init__(
        self, unit: U, period: TimeUnit
    ) -> None:
        base = (
            None
            if unit.is_base_unit() and period.is_base_unit()
            else LinearVelocityUnit.combine(
                unit.get_base_unit(), period.get_base_unit()
            )
        )
        super().__init__(base, unit, period)

    @staticmethod
    def combine(
        numerator: DistanceUnit, denominator: TimeUnit
    ) -> LinearVelocityUnit:
        if not isinstance(numerator, DistanceUnit):
            raise TypeError(
                "Numerator must be a distance unit in a LinearVelocityUnit"
            )
        if not isinstance(denominator, TimeUnit):
            raise TypeError(
                "Denominator must be a time unit in a LinearVelocityUnit"
            )
        return LinearVelocityUnit.combine(numerator, denominator)

    def get_base_unit(self) -> LinearVelocityUnit:
        return cast(LinearVelocityUnit, super().get_base_unit())

    def of(self, magnitude: float) -> LinearVelocity:  # type: ignore[override]
        return LinearVelocity(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(  # type: ignore[override]
        self, base_unit_magnitude: float
    ) -> LinearVelocity:
        return LinearVelocity(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )

    def per(
        self,
        other: Unit
    ) -> LinearAccelerationUnit[LinearVelocityUnit[DistanceUnit]]:
        if isinstance(other, TimeUnit):
            return LinearAccelerationUnit.combine(self, other)
        raise NotImplementedError


LinearVelocityCache: CombinatoryUnitCache[
    DistanceUnit, TimeUnit, LinearVelocityUnit
] = CombinatoryUnitCache(LinearVelocityUnit)


class AngularVelocityUnit[U: AngleUnit](PerUnit[U, TimeUnit]):
    def __init__(
        self, numerator: U, denominator: TimeUnit
    ) -> None:
        base = (
            None
            if numerator.is_base_unit() and denominator.is_base_unit()
            else AngularVelocityUnit.combine(
                numerator.get_base_unit(), denominator.get_base_unit()
            )
        )
        super().__init__(base, numerator, denominator)

    @staticmethod
    def combine(
        numerator: AngleUnit, denominator: TimeUnit
    ) -> AngularVelocityUnit:
        if not isinstance(numerator, AngleUnit):
            raise TypeError(
                "Numerator must be dimensionless in a AngularVelocityUnit"
            )
        if not isinstance(denominator, TimeUnit):
            raise TypeError(
                "Denominator must be a time unit in a AngularVelocityUnit"
            )
        return AngularVelocityCache.combine(numerator, denominator)

    def get_base_unit(self) -> AngularVelocityUnit:
        return cast(AngularVelocityUnit, super().get_base_unit())

    def of(  # type: ignore[override]
        self,
        magnitude: float
    ) -> AngularVelocity:
        return AngularVelocity(
            magnitude, self.to_base_units(magnitude), self
        )

    def of_base_units(  # type: ignore[override]
        self, base_unit_magnitude: float
    ) -> AngularVelocity:
        return AngularVelocity(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )

    def per(
        self,
        other: Unit
    ) -> AngularAccelerationUnit[AngularVelocityUnit[AngleUnit]]:
        if isinstance(other, TimeUnit):
            return AngularAccelerationUnit.combine(self, other)
        raise NotImplementedError


AngularVelocityCache: CombinatoryUnitCache[
    AngleUnit, TimeUnit, AngularVelocityUnit
] = CombinatoryUnitCache(AngularVelocityUnit)


class LinearAccelerationUnit[U: LinearVelocityUnit](PerUnit[U, TimeUnit]):
    def __init__(
        self, numerator: U, denominator: TimeUnit
    ) -> None:
        base = (
            None
            if numerator.is_base_unit() and denominator.is_base_unit()
            else LinearAccelerationUnit.combine(
                numerator.get_base_unit(), denominator.get_base_unit()
            )
        )
        super().__init__(base, numerator, denominator)

    @staticmethod
    def combine(
        numerator: LinearVelocityUnit, denominator: TimeUnit
    ) -> LinearAccelerationUnit:
        return LinearAccelerationCache.combine(numerator, denominator)

    def get_base_unit(self) -> LinearAccelerationUnit:
        return cast(LinearAccelerationUnit, super().get_base_unit())

    def of(  # type: ignore[override]
        self,
        magnitude: float
    ) -> LinearAcceleration:
        return LinearAcceleration(
            magnitude, self.to_base_units(magnitude), self
        )

    def of_base_units(  # type: ignore[override]
        self, base_unit_magnitude: float
    ) -> LinearAcceleration:
        return LinearAcceleration(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )


LinearAccelerationCache: CombinatoryUnitCache[
    LinearVelocityUnit, TimeUnit, LinearAccelerationUnit
] = CombinatoryUnitCache(LinearAccelerationUnit)


class AngularAccelerationUnit[U: AngularVelocityUnit](PerUnit[U, TimeUnit]):
    def __init__(
        self, velocity: U, period: TimeUnit
    ) -> None:
        base = (
            None
            if velocity.is_base_unit() and period.is_base_unit()
            else AngularAccelerationUnit.combine(
                velocity.get_base_unit(), period.get_base_unit()
            )
        )
        super().__init__(base, velocity, period)

    @staticmethod
    def combine(
        numerator: AngularVelocityUnit, denominator: TimeUnit
    ) -> AngularAccelerationUnit:
        if not isinstance(numerator, AngularVelocityUnit):
            raise TypeError(
                "Numerator must be an angular velocity unit "
                "in a AngularAccelerationUnit"
            )
        if not isinstance(denominator, TimeUnit):
            raise TypeError(
                "Denominator must be a time unit in a AngularVelocityUnit"
            )
        return AngularAccelerationCache.combine(numerator, denominator)

    def get_base_unit(self) -> AngularAccelerationUnit:
        return cast(AngularAccelerationUnit, super().get_base_unit())

    def of(  # type: ignore[override]
        self,
        magnitude: float
    ) -> AngularAcceleration:
        return AngularAcceleration(
            magnitude, self.to_base_units(magnitude), self
        )

    def of_base_units(  # type: ignore[override]
        self, base_unit_magnitude: float
    ) -> AngularAcceleration:
        return AngularAcceleration(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )


AngularAccelerationCache: CombinatoryUnitCache[
    AngularVelocityUnit, TimeUnit, AngularAccelerationUnit
] = CombinatoryUnitCache(AngularAccelerationUnit)


class VelocityUnit[U: Unit](PerUnit[U, TimeUnit]):
    def __init__(self, numerator: U, denominator: TimeUnit) -> None:
        base = (
            None
            if numerator.is_base_unit() and denominator.is_base_unit()
            else VelocityUnit.combine(
                numerator.get_base_unit(), denominator.get_base_unit()
            )
        )
        super().__init__(base, numerator, denominator)

    @staticmethod
    def combine(numerator: U, denominator: TimeUnit) -> VelocityUnit[U]:
        if not isinstance(denominator, TimeUnit):
            raise TypeError(
                "Denominator must be a time unit in a VelocityUnit"
            )
        return cast(VelocityUnit,
                    VelocityUnitCache.combine(numerator, denominator))

    def get_base_unit(self) -> VelocityUnit[U]:
        return cast(VelocityUnit, super().get_base_unit())

    def of(  # type: ignore[override]
        self,
        magnitude: float
    ) -> Velocity[U]:
        return Velocity(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(  # type: ignore[override]
        self, base_unit_magnitude: float
    ) -> Velocity[U]:
        return Velocity(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )


VelocityUnitCache: CombinatoryUnitCache[
    Unit, TimeUnit, VelocityUnit[Unit]
] = CombinatoryUnitCache(VelocityUnit)


class ForceUnit[A: MassUnit, B: LinearAccelerationUnit](MultUnit[A, B]):
    def __init__(
        self, mass: A, acceleration: B
    ) -> None:
        base = (
            None
            if mass.is_base_unit() and acceleration.is_base_unit()
            else ForceUnit.combine(
                mass.get_base_unit(), acceleration.get_base_unit()
            )
        )
        super().__init__(base, mass, acceleration)

    @staticmethod
    def combine(
        a: MassUnit, b: LinearAccelerationUnit
    ) -> ForceUnit:
        if not isinstance(a, MassUnit):
            raise TypeError(
                "Numerator (a) must be a mass unit in a ForceUnit"
            )
        if not isinstance(b, LinearAccelerationUnit):
            raise TypeError(
                "Denominator (b) must be a linear acceleration unit "
                "in a ForceUnit"
            )
        return ForceCache.combine(a, b)

    def get_base_unit(self) -> ForceUnit:
        return cast(ForceUnit, super().get_base_unit())

    def mult_as_torque(self, distance: DistanceUnit) -> TorqueUnit:
        return TorqueUnit.combine(distance, self)

    def of(  # type: ignore[override]
        self,
        magnitude: float
    ) -> Force:
        return Force(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(  # type: ignore[override]
        self,
        base_unit_magnitude: float
    ) -> Force:
        return Force(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )


ForceCache: CombinatoryUnitCache[
    MassUnit, LinearAccelerationUnit, ForceUnit
] = CombinatoryUnitCache(ForceUnit)


class TorqueUnit[A: DistanceUnit, B: ForceUnit](MultUnit[A, B]):
    def __init__(
        self, distance: A, force: B
    ) -> None:
        base = (
            None
            if distance.is_base_unit() and force.is_base_unit()
            else TorqueUnit.combine(
                distance.get_base_unit(), force.get_base_unit()
            )
        )
        super().__init__(base, distance, force)

    @staticmethod
    def combine(
        a: DistanceUnit, b: ForceUnit
    ) -> TorqueUnit:
        if not isinstance(a, DistanceUnit):
            raise TypeError(
                "Numerator (a) must be a distance unit in a TorqueUnit"
            )
        if not isinstance(b, ForceUnit):
            raise TypeError(
                "Denominator (b) must be a force unit "
                "in a TorqueUnit"
            )
        return TorqueCache.combine(a, b)

    def get_base_unit(self) -> TorqueUnit:
        return cast(TorqueUnit, super().get_base_unit())

    def of(  # type: ignore[override]
        self,
        magnitude: float
    ) -> Torque:
        return Torque(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(  # type: ignore[override]
        self,
        base_unit_magnitude: float
    ) -> Torque:
        return Torque(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )


TorqueCache: CombinatoryUnitCache[
    DistanceUnit, ForceUnit, TorqueUnit
] = CombinatoryUnitCache(TorqueUnit)


class LinearMomentumUnit[A: MassUnit, B: LinearVelocityUnit](MultUnit[A, B]):
    def __init__(
        self, mass: A, velocity: B
    ) -> None:
        base = (
            None
            if mass.is_base_unit() and velocity.is_base_unit()
            else LinearMomentumUnit.combine(
                mass.get_base_unit(), velocity.get_base_unit()
            )
        )
        super().__init__(base, mass, velocity)

    @staticmethod
    def combine(
        a: MassUnit, b: LinearVelocityUnit
    ) -> LinearMomentumUnit:
        if not isinstance(a, MassUnit):
            raise TypeError(
                "Numerator (a) must be a distance unit in a LinearMomentumUnit"
            )
        if not isinstance(b, LinearVelocityUnit):
            raise TypeError(
                "Denominator (b) must be a force unit "
                "in a LinearMomentumUnit"
            )
        return LinearMomentumCache.combine(a, b)

    def get_base_unit(self) -> LinearMomentumUnit:
        return cast(LinearMomentumUnit, super().get_base_unit())

    def of(  # type: ignore[override]
        self,
        magnitude: float
    ) -> LinearMomentum:
        return LinearMomentum(
            magnitude, self.to_base_units(magnitude), self
        )

    def of_base_units(  # type: ignore[override]
        self, base_unit_magnitude: float
    ) -> LinearMomentum:
        return LinearMomentum(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )

    def mult(  # type: ignore[override]
        self, distance: DistanceUnit
    ) -> AngularMomentumUnit:
        return AngularMomentumUnit.combine(self, distance)


LinearMomentumCache: CombinatoryUnitCache[
    MassUnit, LinearVelocityUnit, LinearMomentumUnit
] = CombinatoryUnitCache(LinearMomentumUnit)


class AngularMomentumUnit[
    A: LinearMomentumUnit,
    B: DistanceUnit
](MultUnit[A, B]):
    def __init__(
        self, momentum: A, distance: B
    ) -> None:
        base = (
            None
            if momentum.is_base_unit() and distance.is_base_unit()
            else AngularMomentumUnit.combine(
                momentum.get_base_unit(), distance.get_base_unit()
            )
        )
        super().__init__(base, momentum, distance)

    @staticmethod
    def combine(
        a: LinearMomentumUnit, b: DistanceUnit
    ) -> AngularMomentumUnit:
        if not isinstance(a, LinearMomentumUnit):
            raise TypeError(
                "Numerator (a) must be a linear momentum unit "
                "in an AngularMomentumUnit"
            )
        if not isinstance(b, DistanceUnit):
            raise TypeError(
                "Denominator (b) must be a distance unit "
                "in an AngularMomentumUnit"
            )
        return AngularMomentumCache.combine(a, b)

    def get_base_unit(self) -> AngularMomentumUnit:
        return cast(AngularMomentumUnit, super().get_base_unit())

    def of(  # type: ignore[override]
        self,
        magnitude: float
    ) -> AngularMomentum:
        return AngularMomentum(
            magnitude, self.to_base_units(magnitude), self
        )

    def of_base_units(  # type: ignore[override]
        self, base_unit_magnitude: float
    ) -> AngularMomentum:
        return AngularMomentum(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )

    def per(  # type: ignore[override]
        self,
        other: AngularVelocityUnit
    ) -> MomentOfInertiaUnit:
        if isinstance(other, AngularVelocityUnit):
            return MomentOfInertiaUnit.combine(self, other)
        raise NotImplementedError


AngularMomentumCache: CombinatoryUnitCache[
    LinearMomentumUnit, DistanceUnit, AngularMomentumUnit
] = CombinatoryUnitCache(AngularMomentumUnit)


class MomentOfInertiaUnit[
    A: AngularMomentumUnit,
    B: AngularVelocityUnit
](PerUnit[A, B]):
    def __init__(
        self,
        numerator: A,
        denominator: B,
    ) -> None:
        base = (
            None
            if numerator.is_base_unit() and denominator.is_base_unit()
            else MomentOfInertiaUnit.combine(
                numerator.get_base_unit(), denominator.get_base_unit()
            )
        )
        super().__init__(base, numerator, denominator)

    @staticmethod
    def combine(
        numerator: AngularMomentumUnit,
        denominator: AngularVelocityUnit
    ) -> MomentOfInertiaUnit:
        if not isinstance(numerator, AngularMomentumUnit):
            raise TypeError(
                "Numerator must be an angular momentum unit "
                "in a MomentOfInteriaUnit"
            )
        if not isinstance(denominator, AngularVelocityUnit):
            raise TypeError(
                "Denominator must be an angular velocity unit "
                "in a MomentOfInteriaUnit"
            )
        return MomentOfInertiaCache.combine(numerator, denominator)

    def get_base_unit(self) -> MomentOfInertiaUnit:
        return cast(MomentOfInertiaUnit, super().get_base_unit())

    def of(  # type: ignore[override]
        self,
        magnitude: float
    ) -> MomentOfInertia:
        return MomentOfInertia(
            magnitude, self.to_base_units(magnitude), self
        )

    def of_base_units(  # type: ignore[override]
        self, base_unit_magnitude: float
    ) -> MomentOfInertia:
        return MomentOfInertia(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )


MomentOfInertiaCache: CombinatoryUnitCache[
    AngularMomentumUnit, AngularVelocityUnit, MomentOfInertiaUnit
] = CombinatoryUnitCache(MomentOfInertiaUnit)


class PowerUnit[
    A: (EnergyUnit
        | VoltageUnit | CurrentUnit
        | AngularVelocityUnit | TorqueUnit),
    B: (TimeUnit
        | CurrentUnit | VoltageUnit
        | TorqueUnit | AngularVelocityUnit)
](PerUnit[A, B]):
    def __init__(self, energy: A, time: B) -> None:
        base = (
            None
            if energy.is_base_unit() and time.is_base_unit()
            else PowerUnit.combine(
                energy.get_base_unit(), time.get_base_unit()
            )
        )
        super().__init__(base, energy, time)

    @staticmethod
    def combine(
        numerator: (
            EnergyUnit
            | VoltageUnit | CurrentUnit
            | AngularVelocityUnit | TorqueUnit
        ),
        denominator: (
            TimeUnit
            | CurrentUnit | VoltageUnit
            | TorqueUnit | AngularVelocityUnit
        )
    ) -> PowerUnit:
        ALLOWED_PAIRS: tuple[tuple[type[Unit], type[Unit]], ...] = (
            (EnergyUnit, TimeUnit),
            (VoltageUnit, CurrentUnit),
            (CurrentUnit, VoltageUnit),
            (AngularVelocityUnit, TorqueUnit),
            (TorqueUnit, AngularVelocityUnit)
        )
        try:
            type_i = next(
                (i for i, (t, _) in enumerate(ALLOWED_PAIRS)
                 if t == type(numerator))
            )
            expect_denom_type = ALLOWED_PAIRS[type_i][1]
            if not isinstance(denominator, expect_denom_type):
                raise TypeError(
                    "Second element does not have correct matching type:"
                    f"{type(denominator)} is invalid, "
                    f"expected {expect_denom_type}")
            return PowerCache.combine(numerator, denominator)
        except (StopIteration, IndexError):
            raise TypeError("Type not found")

    def get_base_unit(self) -> PowerUnit:
        return cast(PowerUnit, super().get_base_unit())

    def of(  # type: ignore[override]
        self,
        magnitude: float
    ) -> Power:
        return Power(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(  # type: ignore[override]
        self,
        base_unit_magnitude: float
    ) -> Power:
        return Power(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )


PowerCache: CombinatoryUnitCache[
    (
        EnergyUnit
        | VoltageUnit | CurrentUnit
        | AngularVelocityUnit | TorqueUnit
    ),
    (
        TimeUnit
        | CurrentUnit | VoltageUnit
        | TorqueUnit | AngularVelocityUnit
    ),
    PowerUnit
] = CombinatoryUnitCache(PowerUnit)


class ResistanceUnit[A: VoltageUnit, B: CurrentUnit](PerUnit[A, B]):
    def __init__(
        self, numerator: A, denominator: B
    ) -> None:
        base = (
            None
            if numerator.is_base_unit() and denominator.is_base_unit()
            else ResistanceUnit.combine(
                numerator.get_base_unit(), denominator.get_base_unit()
            )
        )
        super().__init__(base, numerator, denominator)

    @staticmethod
    def combine(
        numerator: VoltageUnit,
        denominator: CurrentUnit
    ) -> ResistanceUnit:
        if not isinstance(numerator, VoltageUnit):
            raise TypeError(
                "Numerator must be a voltage unit in a ResistanceUnit"
            )
        if not isinstance(denominator, CurrentUnit):
            raise TypeError(
                "Denominator must be a current unit "
                "in a ResistanceUnit"
            )
        return ResistanceCache.combine(numerator, denominator)

    def get_base_unit(self) -> ResistanceUnit:
        return cast(ResistanceUnit, super().get_base_unit())

    def of(self, magnitude: float) -> Resistance:  # type: ignore[override]
        return Resistance(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(  # type: ignore[override]
        self, base_unit_magnitude: float
    ) -> Resistance:
        return Resistance(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )


ResistanceCache: CombinatoryUnitCache[
    VoltageUnit, CurrentUnit, ResistanceUnit
] = CombinatoryUnitCache(ResistanceUnit)


class BaseUnits:
    DistanceUnit: Final = DistanceUnit(None, 1.0, "Meter", "m")  # noqa: S116
    TimeUnit: Final = TimeUnit(None, 1.0, "Second", "s")  # noqa: S116
    MassUnit: Final = MassUnit(None, 1.0, "Kilogram", "Kg")  # noqa: S116
    AngleUnit: Final = AngleUnit(None, 1.0, "Radian", "rad")  # noqa: S116
    VoltageUnit: Final = VoltageUnit(None, 1.0, "Volt", "V")  # noqa: S116
    CurrentUnit: Final = CurrentUnit(None, 1.0, "Amp", "A")  # noqa: S116
    EnergyUnit: Final = EnergyUnit(None, 1.0, "Joule", "J")  # noqa: S116
    TemperatureUnit: Final = TemperatureUnit(  # noqa: S116
        None, UnaryFunction.IDENTITY, UnaryFunction.IDENTITY, "Kelvin", "K"
    )
    Value: Final = DimensionlessUnit(None, 1.0, "<?>", "<?>")  # noqa: S116


class Units:
    Value: Final = BaseUnits.Value  # noqa: S116
    Percent: Final = (UnitBuilder(Value).split_into(100.0)  # noqa: S116
                      .named("Percent").symbol("%")
                      .make_and_cast(DimensionlessUnit))

    Meters: Final = BaseUnits.DistanceUnit  # noqa: S116
    Meter: Final = Meters  # noqa: S116
    Millimeters: Final = (UnitBuilder(Meters).split_into(1000.0)  # noqa: S116
                          .named("Millimeter").symbol("mm")
                          .make_and_cast(DistanceUnit))
    Millimeter: Final = Millimeters  # noqa: S116
    Centimeters: Final = (UnitBuilder(Meters).split_into(100.0)  # noqa: S116
                          .named("Centimeter").symbol("cm")
                          .make_and_cast(DistanceUnit))
    Centimeter: Final = Centimeters  # noqa: S116
    Inches: Final = (UnitBuilder(Millimeters).aggregate(25.4)  # noqa: S116
                     .named("Inch").symbol("in")
                     .make_and_cast(DistanceUnit))
    Inch: Final = Inches  # noqa: S116
    Feet: Final = (UnitBuilder(Inches).aggregate(12.0)  # noqa: S116
                   .named("Foot").symbol("ft")
                   .make_and_cast(DistanceUnit))
    Foot: Final = Feet  # noqa: S116

    Seconds: Final = BaseUnits.TimeUnit  # noqa: S116
    Second: Final = Seconds  # noqa: S116
    Milliseconds: Final = (UnitBuilder(Seconds)  # noqa: S116
                           .split_into(1000.0).named("Millisecond")
                           .symbol("ms")
                           .make_and_cast(TimeUnit))
    Millisecond: Final = Milliseconds  # noqa: S116
    Microseconds: Final = (UnitBuilder(Seconds)  # noqa: S116
                           .split_into(1_000_000.0).named("Microsecond")
                           .symbol("us")
                           .make_and_cast(TimeUnit))
    Microsecond: Final = Microseconds  # noqa: S116
    Nanoseconds: Final = (UnitBuilder(Seconds)  # noqa: S116
                          .split_into(1_000_000_000.0).named("Nanosecond")
                          .symbol("ns")
                          .make_and_cast(TimeUnit))
    Nanosecond: Final = Nanoseconds  # noqa: S116
    Minutes: Final = (UnitBuilder(Seconds)  # noqa: S116
                      .aggregate(60.0).named("Minute")
                      .symbol("min")
                      .make_and_cast(TimeUnit))
    Minute: Final = Minutes  # noqa: S116

    Radians: Final = BaseUnits.AngleUnit  # noqa: S116
    Radian: Final = Radians  # noqa: S116
    Revolutions: Final = (UnitBuilder(Radians)  # noqa: S116
                          .aggregate(2.0 * math.pi)
                          .named("Revolution").symbol("R")
                          .make_and_cast(AngleUnit))
    Revolution: Final = Revolutions  # noqa: S116
    Rotations: Final = Revolutions  # noqa: S116
    Rotation: Final = Rotations  # noqa: S116
    Degrees: Final = (UnitBuilder(Revolutions).split_into(360.0)  # noqa: S116
                      .named("Degree").symbol("\xb0")
                      .make_and_cast(AngleUnit))
    Degree: Final = Degrees  # noqa: S116

    MetersPerSecond: Final = Meters.per(Second)  # noqa: S116
    FeetPerSecond: Final = Feet.per(Second)  # noqa: S116
    InchesPerSecond: Final = Inches.per(Second)  # noqa: S116
    RevolutionsPerSecond: Final = Revolutions.per(Second)  # noqa: S116
    RotationsPerSecond: Final = Rotations.per(Second)  # noqa: S116
    RotationsPerMinute: Final = Rotations.per(Minute)  # noqa: S116
    RPM: Final = RotationsPerMinute  # noqa: S116
    RadiansPerSecond: Final = Radians.per(Second)  # noqa: S116
    DegreesPerSecond: Final = Degrees.per(Second)  # noqa: S116

    Hertz: Final = (UnitBuilder(Value.per(Second))  # noqa: S116
                    .named("Hertz").symbol("Hz")
                    .make_and_cast(FrequencyUnit[DimensionlessUnit]))
    Millihertz: Final = (UnitBuilder(Hertz).split_into(1000.0)  # noqa: S116
                         .named("Millihertz").symbol("MHz")
                         .make_and_cast(FrequencyUnit[DimensionlessUnit]))

    MetersPerSecondPerSecond: Final = MetersPerSecond.per(Second)  # noqa: S116
    FeetPerSecondPerSecond: Final = FeetPerSecond.per(Second)  # noqa: S116
    InchesPerSecondPerSecond: Final = InchesPerSecond.per(Second)  # noqa: S116
    RotationsPerSecondPerSecond: Final = (  # noqa: S116
        RotationsPerSecond.per(Second))
    RadiansPerSecondPerSecond: Final = (  # noqa: S116
        RadiansPerSecond.per(Second))
    DegreesPerSecondPerSecond: Final = (  # noqa: S116
        DegreesPerSecond.per(Second))
    Gs: Final = (UnitBuilder(MetersPerSecondPerSecond)  # noqa: S116
                 .aggregate(9.80665)
                 .named("G").symbol("G")
                 .make_and_cast(LinearAccelerationUnit[
                     LinearVelocityUnit[DistanceUnit]
                     ]))

    Kilograms: Final = BaseUnits.MassUnit  # noqa: S116
    Kilogram: Final = Kilograms  # noqa: S116
    Grams: Final = (UnitBuilder(Kilograms).split_into(1000.0)  # noqa: S116
                    .named("Gram").symbol("g")
                    .make_and_cast(MassUnit))
    Gram: Final = Grams  # noqa: S116
    Pounds: Final = (UnitBuilder(Grams).aggregate(453.592)  # noqa: S116
                     .named("Pound").symbol("lb.")
                     .make_and_cast(MassUnit))
    Pound: Final = Pounds  # noqa: S116
    Ounces: Final = (UnitBuilder(Pounds).split_into(16.0)  # noqa: S116
                     .named("Ounce").symbol("oz.")
                     .make_and_cast(MassUnit))
    Ounce: Final = Ounces  # noqa: S116

    Newtons: Final = (  # noqa: S116
        UnitBuilder(Kilograms.mult(MetersPerSecondPerSecond))
        .named("Newton").symbol("N")
        .make_and_cast(ForceUnit[
                            MassUnit,
                            LinearAccelerationUnit[
                                LinearVelocityUnit[DistanceUnit]
                            ]
                        ]))
    Newton: Final = Newtons  # noqa: S116
    PoundsForce: Final = (UnitBuilder(Pounds.mult(Gs))  # noqa: S116
                          .named("Pound-force").symbol("lbsf.")
                          .make_and_cast(ForceUnit[
                              MassUnit,
                              LinearAccelerationUnit[
                                LinearVelocityUnit[DistanceUnit]
                              ]
                          ]))
    PoundForce: Final = PoundsForce  # noqa: S116
    OuncesForce: Final = (UnitBuilder(Ounces.mult(Gs))  # noqa: S116
                          .named("Ounce-force").symbol("ozf")
                          .make_and_cast(ForceUnit[
                              MassUnit,
                              LinearAccelerationUnit[
                                LinearVelocityUnit[DistanceUnit]
                              ]
                          ]))
    OunceForce: Final = OuncesForce  # noqa: S116

    NewtonMeters: Final = Meters.mult_as_torque(Newtons)  # noqa: S116
    NewtonMeter: Final = NewtonMeters  # noqa: S116
    PoundFeet: Final = Feet.mult_as_torque(PoundsForce)  # noqa: S116
    PoundFoot: Final = PoundFeet  # noqa: S116
    PoundInches: Final = Inches.mult_as_torque(PoundsForce)  # noqa: S116
    PoundInch: Final = PoundInches  # noqa: S116
    OunceInches: Final = Inches.mult_as_torque(OuncesForce)  # noqa: S116
    OunceInch: Final = OunceInches  # noqa: S116

    KilogramMetersPerSecond: Final = (  # noqa: S116
        Kilograms.mult(MetersPerSecond).cast(MultUnit[
            MassUnit,
            LinearVelocityUnit[DistanceUnit]
        ])
    )
    PoundInchesPerSecond: Final = (  # noqa: S116
        Pounds.mult(InchesPerSecond).cast(MultUnit[
            MassUnit,
            LinearVelocityUnit[DistanceUnit]
        ])
    )

    KilogramMetersSquaredPerSecond: Final = (  # noqa: S116
        KilogramMetersPerSecond.mult(Meters).cast(MultUnit[
            MassUnit,
            MultUnit[
                MassUnit,
                LinearVelocityUnit[DistanceUnit]
            ]
        ])
    )
    PoundInchesSquaredPerSecond: Final = (  # noqa: S116
        PoundInchesPerSecond.mult(Inches).cast(MultUnit[
            MassUnit,
            MultUnit[
                MassUnit,
                LinearVelocityUnit[DistanceUnit]
            ]
        ])
    )

    KilogramSquareMeters: Final = (  # noqa: S116
        KilogramMetersSquaredPerSecond.per(RadiansPerSecond).cast(PerUnit[
            MultUnit[
                MassUnit,
                MultUnit[
                    MassUnit,
                    LinearVelocityUnit[DistanceUnit]
                ]
            ],
            AngularVelocityUnit[AngleUnit]
        ])
    )
    PoundSquareInches: Final = (  # noqa: S116
        PoundInchesSquaredPerSecond.per(RadiansPerSecond).cast(PerUnit[
            MultUnit[
                MassUnit,
                MultUnit[
                    MassUnit,
                    LinearVelocityUnit[DistanceUnit]
                ]
            ],
            AngularVelocityUnit[AngleUnit]
        ])
    )

    Volts: Final = BaseUnits.VoltageUnit  # noqa: S116
    Volt: Final = Volts  # noqa: S116
    Millivolts: Final = (UnitBuilder(Volts).split_into(1000.0)  # noqa: S116
                         .named("Millivolt").symbol("mV")
                         .make_and_cast(VoltageUnit))
    Millivolt: Final = Millivolts  # noqa: S116

    Amps: Final = BaseUnits.CurrentUnit  # noqa: S116
    Amp: Final = Amps  # noqa: S116
    Milliamps: Final = (UnitBuilder(Amps).split_into(1000.0)  # noqa: S116
                        .named("Milliamp").symbol("mA")
                        .make_and_cast(CurrentUnit))
    Milliamp: Final = Milliamps  # noqa: S116

    Ohms: Final = (UnitBuilder(Volts.per(Amp))  # noqa: S116
                   .named("Ohm").symbol("\u03a9")
                   .make_and_cast(ResistanceUnit))
    Ohm: Final = Ohms  # noqa: S116
    KiloOhms: Final = (UnitBuilder(Ohms).aggregate(1000.0)  # noqa: S116
                       .named("KiloOhm").symbol("k\u03a9")
                       .make_and_cast(ResistanceUnit))
    KiloOhm: Final = KiloOhms  # noqa: S116
    MilliOhms: Final = (UnitBuilder(Ohms).split_into(1000.0)  # noqa: S116
                        .named("MilliOhm").symbol("m\u03a9")
                        .make_and_cast(ResistanceUnit))
    MilliOhm: Final = MilliOhms  # noqa: S116

    Joules: Final = BaseUnits.EnergyUnit  # noqa: S116
    Joule: Final = Joules  # noqa: S116
    Millijoules: Final = (UnitBuilder(Joules).split_into(1000.0)  # noqa: S116
                          .named("Millijoule").symbol("mJ")
                          .make_and_cast(EnergyUnit))
    Millijoule: Final = Millijoules  # noqa: S116
    Kilojoules: Final = (UnitBuilder(Joules).aggregate(1000.0)  # noqa: S116
                         .named("Kilojoule").symbol("kJ")
                         .make_and_cast(EnergyUnit))
    Kilojoule: Final = Kilojoules  # noqa: S116

    Watts: Final = (UnitBuilder(Joules.per(Second))  # noqa: S116
                    .named("Watt").symbol("W")
                    .make_and_cast(PerUnit[EnergyUnit, TimeUnit]))
    Watt: Final = Watts  # noqa: S116
    Milliwatts: Final = (UnitBuilder(Watts).split_into(1000.0)  # noqa: S116
                         .named("Milliwatt").symbol("mW")
                         .make_and_cast(PerUnit[EnergyUnit, TimeUnit]))
    Milliwatt: Final = Milliwatts  # noqa: S116
    Horsepower: Final = (UnitBuilder(Watts).aggregate(745.7)  # noqa: S116
                         .named("Horsepower").symbol("HP")
                         .make_and_cast(PerUnit[EnergyUnit, TimeUnit]))

    Kelvin: Final = BaseUnits.TemperatureUnit  # noqa: S116
    Celsius: Final = (UnitBuilder(Kelvin).offset(273.15)  # noqa: S116
                      .named("Celsius").symbol("\xb0C")
                      .make_and_cast(TemperatureUnit))
    Fahrenheit: Final = (UnitBuilder(Celsius)  # noqa: S116
                         .mapping_input_range(0.0, 100.0)
                         .to_output_range(32.0, 212.0)
                         .named("Fahrenheit").symbol("\xb0F")
                         .make_and_cast(TemperatureUnit))

    VoltsPerMeterPerSecond: Final = (Volts.per(MetersPerSecond)  # noqa: S116
                                     .cast(PerUnit[
                                         VoltageUnit,
                                         LinearVelocityUnit[DistanceUnit]
                                     ]))

    VoltsPerMeterPerSecondSquared: Final = (  # noqa: S116
        Volts.per(MetersPerSecondPerSecond).cast(PerUnit[
            VoltageUnit,
            LinearAccelerationUnit[LinearVelocityUnit[DistanceUnit]]
        ])
    )
    VoltsPerRadianPerSecond: Final = (Volts.per(RadiansPerSecond)  # noqa: S116
                                      .cast(PerUnit[
                                          VoltageUnit,
                                          AngularVelocityUnit[AngleUnit]
                                      ]))
    VoltsPerRadianPerSecondSquared: Final = (  # noqa: S116
        Volts.per(RadiansPerSecond.per(Second)).cast(PerUnit[
            VoltageUnit,
            AngularAccelerationUnit[AngularVelocityUnit[AngleUnit]]
        ])
    )

    @staticmethod
    def derive(unit: U) -> UnitBuilder[U]:
        return UnitBuilder(unit)

    @classmethod
    def Milli[U: Unit](  # noqa: S100
        cls,
        base_unit: U,
        name: str | None = None,
        symbol: str | None = None
    ) -> U:
        if name is not None and symbol is not None:
            return (cls.derive(base_unit)
                    .split_into(1000)
                    .named(name).symbol(symbol)
                    .make_and_cast(type(base_unit)))
        elif name is None and symbol is None:
            name_lower = _locale_invariant_lower(base_unit.name())
            symbol_lower = _locale_invariant_lower(base_unit.symbol())
            return (cls.derive(base_unit)
                    .split_into(1000)
                    .named(f"Milli{name_lower}")
                    .symbol(f"m{symbol_lower}")
                    .make_and_cast(type(base_unit)))
        raise ValueError(
            "Milli: Either pass in only a base_unit"
            "or add both a name and a symbol (non-None)"
        )

    @classmethod
    def Micro[U: Unit](  # noqa: S100
        cls,
        base_unit: U,
        name: str | None = None,
        symbol: str | None = None
    ) -> U:
        if name is not None and symbol is not None:
            return (cls.derive(base_unit)
                    .split_into(1_000_000)
                    .named(name).symbol(symbol)
                    .make_and_cast(type(base_unit)))
        elif name is None and symbol is None:
            name_lower = _locale_invariant_lower(base_unit.name())
            symbol_lower = _locale_invariant_lower(base_unit.symbol())
            return (cls.derive(base_unit)
                    .split_into(1_000_000)
                    .named(f"Micro{name_lower}")
                    .symbol(f"u{symbol_lower}")
                    .make_and_cast(type(base_unit)))
        raise ValueError(
            "Micro: Either pass in only a base_unit"
            "or add both a name and a symbol (non-None)"
        )

    @classmethod
    def Kilo[U: Unit](  # noqa: S100
        cls,
        base_unit: U,
        name: str | None = None,
        symbol: str | None = None
    ) -> U:
        if name is not None and symbol is not None:
            return (cls.derive(base_unit)
                    .aggregate(1000)
                    .named(name).symbol(symbol)
                    .make_and_cast(type(base_unit)))
        elif name is None and symbol is None:
            name_lower = _locale_invariant_lower(base_unit.name())
            symbol_lower = _locale_invariant_lower(base_unit.symbol())
            return (cls.derive(base_unit)
                    .aggregate(1000)
                    .named(f"Kilo{name_lower}")
                    .symbol(f"K{symbol_lower}")
                    .make_and_cast(type(base_unit)))
        raise ValueError(
            "Kilo: Either pass in only a base_unit"
            "or add both a name and a symbol (non-None)"
        )
