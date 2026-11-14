# Copyright (c) FIRST and other WPILib contributors.
# Open Source Software; you can modify and/or share it under the terms of
# the WPILib BSD license file in the root directory of this project.
#
# Contributors:
# - WPILib
# - Alan M. Burr <info@alanburr.us.eu.org>

from __future__ import annotations

import copy
import locale
import math
from collections.abc import Callable
from typing import (
    Any,
    Final,
    NoReturn,
    Self,
    TypedDict,
    TypeVar,
    Unpack,
    cast,
    final,
    overload,
)

from .collections import LongToObjectHashMap
from .measure import (
    Acceleration,
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
    try:
        nowlocale = locale.getlocale()
    except TypeError:
        nowlocale = None
    locale.setlocale(locale.LC_ALL, "C")
    try:
        return string.lower()
    finally:
        if nowlocale is not None:
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
        return float(x)

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
        def _div(x: float) -> float:
            n = self.apply(x)
            d = (
                divisor if isinstance(divisor, (float, int))
                else divisor.apply(x))
            if n == 0:
                return 0
            if d == 0:
                # strange div by 0 case but whatever
                return math.inf * math.copysign(1.0, n)
            return n / d
        return UnaryFunction(_div)

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

    class UnitConstructorArgs(TypedDict):
        base_unit: Unit | None
        """ The base unit """
        to_base_converter: UnaryFunction
        """ Used to convert units of this type to the base unit """
        from_base_converter: UnaryFunction
        """ Used to convert base units to units of this type """
        unit_name: str | None
        """ Unit name (singular noun) """
        unit_symbol: str | None
        """ Unit symbol """

    class UnitConstructorArgsUsingMultiplier(TypedDict):
        base_unit: Unit | None
        """ The base unit """
        base_unit_equivalent: float
        """ Multiplier to convert this unit *to* the base unit """
        unit_name: str | None
        """ Unit name (singular noun) """
        unit_symbol: str | None
        """ Unit symbol """

    @overload
    def __init__(self, **kwargs: Unpack[UnitConstructorArgs]):
        ...

    @overload
    def __init__(self, **kwargs: Unpack[UnitConstructorArgsUsingMultiplier]):
        ...

    def __init__(self, **kwargs) -> None:
        base_unit = kwargs["base_unit"]
        to_base_converter = kwargs.get("to_base_converter")
        from_base_converter = kwargs.get("from_base_converter")
        unit_name = kwargs["unit_name"]
        unit_symbol = kwargs["unit_symbol"]
        base_equivalent = kwargs.get("base_unit_equivalent")

        self._base_unit = base_unit if base_unit is not None else self

        if (
            base_equivalent is not None
            and (
                to_base_converter is not None
                or from_base_converter is not None
            )
        ):
            raise TypeError(
                "base_equivalent cannot be used with "
                "to/from_base_converter"
            )

        self._to_base_converter: UnaryFunction
        self._from_base_converter: UnaryFunction

        if isinstance(to_base_converter, UnaryFunction):
            if not isinstance(from_base_converter, UnaryFunction):
                raise TypeError(
                    "from_base_converter must be a UnaryFunction|float."
                )

            self._to_base_converter = to_base_converter
            self._from_base_converter = from_base_converter
        elif isinstance(base_equivalent, (float, int)):
            equiv = float(base_equivalent)
            self._to_base_converter = UnaryFunction(lambda x: x * equiv)
            self._from_base_converter = UnaryFunction(lambda x: x / equiv)

        if unit_name is None:
            raise ValueError("unit_name must not be None.")
        if unit_symbol is None:
            raise ValueError("unit_symbol must not be None.")

        self._name: str = unit_name
        self._symbol: str = unit_symbol

        self._zero: Measure = self.of(0.0)
        self._one: Measure = self.of(1.0)

    def _raiseTypeErrorPer(self, other_type: Any) -> NoReturn:  # noqa: S100
        raise TypeError(f"per: other must be a unit, got {type(other_type)}")

    def of(self, magnitude: float) -> ImmutableMeasure[Self] | Per[Unit, Unit]:
        raise NotImplementedError("Unit.of()")

    def of_base_units(
        self,
        base_unit_magnitude: float
    ) -> ImmutableMeasure[Self] | Per[Unit, Unit]:
        raise NotImplementedError("Unit.of_base_unit()")

    def zero(self) -> Measure[Self] | ImmutableMeasure[Self]:
        return self._zero

    def one(self) -> Measure[Self] | ImmutableMeasure[Self]:
        return self._one

    def per[U: Unit](self, other: U) -> PerUnit[Self, U]:
        if not isinstance(other, Unit):
            self._raiseTypeErrorPer(other)
        return PerUnit.combine(self, other)

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
        if not isinstance(other, Unit):
            return False
        if self is other:
            return True
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

    def mult[O: Unit](self, other: O) -> MultUnit[Self, O]:
        raise NotImplementedError("Unit.mult()")

    def __eq__(self, o: object) -> bool:
        return self is o or (
            isinstance(o, Unit)
            and self._name == o._name
            and self._symbol == o._symbol
            and self.equivalent(o)
        ) or (
            isinstance(o, Unit)
            and Unit._unit_signature(o) == Unit._unit_signature(self) and
            self.equivalent(o)
        )

    def __hash__(self) -> int:
        return hash((
            id(self._to_base_converter),
            id(self._from_base_converter),
            self._name,
            self._symbol,
        ))

    def name(self) -> str:
        return self._name.strip()

    def symbol(self) -> str:
        return self._symbol.strip()

    def cast[U: Unit](self, typecast: type[U]) -> U:
        return cast(typecast, self)  # type: ignore

    def __str__(self) -> str:
        return self.name()

    @classmethod
    def _unit_signature(cls, other: Unit) -> dict[type[Unit], int]:
        if isinstance(other, MultUnit):
            result = cls._unit_signature(other.unit_a())
            for atom, power in cls._unit_signature(other.unit_b()).items():
                result[atom] = result.get(atom, 0) + power
            return {atom: power for atom, power in result.items() if power}

        if isinstance(other, PerUnit):
            result = cls._unit_signature(other.numerator())
            for atom, power in (
                cls._unit_signature(other.denominator()).items()
            ):
                result[atom] = result.get(atom, 0) - power
            return {atom: power for atom, power in result.items() if power}

        return {type(other): 1}


class CombinatoryUnitCache[A, B, Out]:
    """A helper class for creating and caching combined units."""

    def __init__(
        self,
        constructor: Callable[[A, B], Out]
    ) -> None:
        self._cache: LongToObjectHashMap[Out] = LongToObjectHashMap()
        self._constructor = constructor

    def combine(self, a: A, b: B) -> Out:
        if not isinstance(a, Unit):
            raise TypeError(
                "CombinatoryUnitCache.combine: a must be a unit, "
                f"got {type(a)}"
            )
        if not isinstance(b, Unit):
            raise TypeError(
                "CombinatoryUnitCache.combine: b must be a unit, "
                f"got {type(b)}"
            )

        key = (hash(a) << 32) | (hash(b) & 0xFFFFFFFF)
        existing = self._cache.get(key)
        if existing is not None:
            return existing

        new_unit = self._constructor(a, b)  # type: ignore[arg-type]
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

    def offset(self, offset_val: float) -> Self:
        self._m_to_base = UnaryFunction(lambda x: x + float(offset_val))
        self._m_from_base = UnaryFunction(lambda x: x - float(offset_val))
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
                    (float(v) - float(in_min))
                    * (float(out_max) - float(out_min))
                    / (float(in_max) - float(in_min))
                    + float(out_min)
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

    def from_base(self, from_base: UnaryFunction) -> Self:
        self._m_from_base = from_base
        return self

    def to_base(self, to_base: UnaryFunction) -> Self:
        self._m_to_base = to_base
        return self

    def named(self, name: str) -> Self:
        self._m_name = name
        return self

    def symbol(self, symbol: str) -> Self:
        self._m_symbol = symbol
        return self

    def split_into(self, fraction: float) -> UnitBuilder[A]:
        if fraction == 0:
            raise ValueError("Fraction must be nonzero")
        return self.to_base(
            UnaryFunction(lambda x: x / float(fraction))
        ).from_base(
            UnaryFunction(lambda b: b * float(fraction))
        )

    def aggregate(self, aggregation: float) -> UnitBuilder[A]:
        if aggregation == 0:
            raise ValueError("Aggregation amount must be nonzero")
        return self.to_base(
            UnaryFunction(lambda x: x * float(aggregation))
        ).from_base(
            UnaryFunction(lambda b: b / float(aggregation))
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

        built: Unit
        if isinstance(self._m_base, PerUnit):
            if type(self._m_base) is PerUnit:
                built = PerUnit(None, self._m_base.numerator(),
                                self._m_base.denominator())
            else:
                built = type(self._m_base)(  # type: ignore
                    self._m_base.numerator(),
                    self._m_base.denominator()
                )
            built._to_base_converter = to_base
            built._from_base_converter = from_base
            built._name = self._m_name
            built._symbol = self._m_symbol
            built._zero = built.of(0.0)
            built._one = built.of(1.0)
            return built

        if isinstance(self._m_base, MultUnit):
            if type(self._m_base) is MultUnit:
                built = MultUnit(None,
                                 self._m_base.unit_a(),
                                 self._m_base.unit_b())
            else:
                built = type(self._m_base)(  # type: ignore
                    self._m_base.unit_a(),
                    self._m_base.unit_b()
                )
            built._to_base_converter = to_base
            built._from_base_converter = from_base
            built._name = self._m_name
            built._symbol = self._m_symbol
            built._zero = built.of(0.0)
            built._one = built.of(1.0)
            return built

        built = copy.copy(self._m_base)
        Unit.__init__(
            built,
            base_unit=base_unit,
            to_base_converter=to_base,
            from_base_converter=from_base,
            unit_name=self._m_name,
            unit_symbol=self._m_symbol
        )
        return built

    def make_and_cast[T: Unit](
        self,
        typecast: type[T],
        constructor: Callable[
            [Unit, UnaryFunction, UnaryFunction, str, str], Unit
        ] | None = None,
    ) -> T:
        return cast(typecast, self.make(constructor))  # type: ignore


class DimensionlessUnit(Unit):
    def get_base_unit(self) -> Self:
        return super().get_base_unit()

    def of(self, magnitude: float) -> Dimensionless:  # type: ignore
        return Dimensionless(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(  # type: ignore
        self,
        base_unit_magnitude: float
    ) -> Dimensionless:
        return Dimensionless(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )

    def zero(self) -> Dimensionless:  # type: ignore
        return cast(Dimensionless, super().zero())

    def one(self) -> Dimensionless:  # type: ignore
        return cast(Dimensionless, super().one())

    def convert_from(
        self,
        magnitude: float,
        other_unit: DimensionlessUnit
    ) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))

    @overload
    def per(self, other: TimeUnit) -> FrequencyUnit: ...

    @overload
    def per[O: Unit](self, unit: O) -> PerUnit[Self, O]: ...

    def per(  # type: ignore
        self, other
    ):
        if isinstance(other, TimeUnit):
            return cast(FrequencyUnit, FrequencyUnit.combine(self, other))
        return super().per(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[Self, U]:
        return PerUnit.combine(self, other)


class TimeUnit(Unit):
    def get_base_unit(self) -> TimeUnit:
        return super().get_base_unit()

    @overload
    def per(self, other: TimeUnit) -> DimensionlessUnit: ...  # type: ignore

    @overload
    def per[O: Unit](self, other: O) -> PerUnit[Self, O]: ...

    def per(  # type: ignore
        self, other
    ):
        if isinstance(other, TimeUnit):
            return Units.derive(Units.Value).to_base(
                self.get_converter_to_base().div(
                    other.get_converter_to_base())
            ).from_base(
                other.get_converter_from_base().div(
                    self.get_converter_from_base())
            ).named(self.name() + " per " + other.name()).symbol(
                self.symbol() + "/" + other.symbol()
            ).make()
        return super().per(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[Self, U]:
        return PerUnit.combine(self, other)

    def convert_from(self, magnitude: float, other_unit: TimeUnit) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))

    def of(self, magnitude: float) -> Time:  # type: ignore
        return Time(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(  # type: ignore
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
    def get_base_unit(self) -> DistanceUnit:
        return super().get_base_unit()

    @overload
    def per(self, other: TimeUnit) -> LinearVelocityUnit[Self]: ...

    @overload
    def per[O: Unit](self, other: O) -> PerUnit[Self, O]: ...

    def per(  # type: ignore
        self, other
    ):
        if isinstance(other, TimeUnit):
            return LinearVelocityUnit.combine(self, other)
        return self.per_unit(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[Self, U]:
        return PerUnit.combine(self, other)

    def convert_from(
        self,
        magnitude: float,
        other_unit: DistanceUnit
    ) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))

    def of(self, magnitude: float) -> Distance:  # type: ignore
        return Distance(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(  # type: ignore
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
        Self,
        ForceUnit[
            MassUnit,
            LinearAccelerationUnit[LinearVelocityUnit[Self]]
        ]
    ]:
        return TorqueUnit.combine(self, force)


class MassUnit(Unit):
    def get_base_unit(self) -> MassUnit:
        return super().get_base_unit()

    def of(self, magnitude: float) -> Mass:  # type: ignore
        return Mass(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(  # type: ignore
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

    @overload
    def per(self, other: TimeUnit) -> VelocityUnit[Self]: ...

    @overload
    def per[O: Unit](self, other: O) -> PerUnit[Self, O]: ...

    def per(self, other):  # type: ignore
        if isinstance(other, TimeUnit):
            return VelocityUnit.combine(self, other)
        return super().per(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[MassUnit, U]:
        return PerUnit.combine(self, other)

    def convert_from(self, magnitude: float, other_unit: MassUnit) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))

    @overload
    def mult[O: LinearVelocityUnit](
        self, other: O
    ) -> LinearMomentumUnit[Self, O]: ...

    @overload
    def mult[O: LinearAccelerationUnit](
        self, other: O
    ) -> ForceUnit[Self, O]: ...

    @overload
    def mult[O: Unit](self, other: O) -> MultUnit[Self, O]: ...

    def mult(self, other):  # type: ignore
        if isinstance(other, LinearVelocityUnit):
            return LinearMomentumUnit.combine(self, other)
        elif isinstance(other, LinearAccelerationUnit):
            return ForceUnit.combine(self, other)
        raise TypeError(f"Invalid argument for mult: {type(other)}")


class AngleUnit(Unit):
    def get_base_unit(self) -> AngleUnit:
        return super().get_base_unit()

    def of(self, magnitude: float) -> Angle:  # type: ignore
        return Angle.of_relative_units(magnitude, self)

    def of_base_units(  # type: ignore
        self,
        base_unit_magnitude: float
    ) -> Angle:
        return Angle.of_base_units(base_unit_magnitude, self)

    @overload
    def per(self, other: TimeUnit) -> AngularVelocityUnit[Self]: ...

    @overload
    def per[O: Unit](self, other: O) -> PerUnit[Self, O]: ...

    def per(  # type: ignore
        self, other
    ):
        if isinstance(other, TimeUnit):
            return AngularVelocityUnit.combine(self, other)
        return super().per(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[Self, U]:
        return PerUnit.combine(self, other)

    def convert_from(self, magnitude: float, other_unit: AngleUnit) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))

    def zero(self) -> Angle:
        return cast(Angle, super().zero())

    def one(self) -> Angle:
        return cast(Angle, super().one())


class VoltageUnit(Unit):
    def get_base_unit(self) -> VoltageUnit:
        return super().get_base_unit()

    def mult[O: CurrentUnit](  # type: ignore[override]
        self, current: O, name: str, symbol: str  # noqa: S2638
    ) -> PowerUnit[Self, O]:
        return cast(PowerUnit, Units.derive(
            PowerUnit.combine(self, current)
        ).named(name).symbol(symbol).make())

    def of(self, magnitude: float) -> Voltage:  # type: ignore
        return Voltage(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(  # type: ignore
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

    @overload
    def per(self, other: CurrentUnit) -> ResistanceUnit: ...

    @overload
    def per(self, other: TimeUnit) -> VelocityUnit[Self]: ...

    @overload
    def per[O: Unit](self, other: O) -> PerUnit[Self, O]: ...

    def per(self, other):  # type: ignore
        if isinstance(other, CurrentUnit):
            return ResistanceUnit.combine(self, other)
        elif isinstance(other, TimeUnit):
            return VelocityUnit.combine(self, other)
        return super().per(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[VoltageUnit, U]:
        return PerUnit.combine(self, other)

    def convert_from(self, magnitude: float, other_unit: VoltageUnit) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))


class CurrentUnit(Unit):
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
        if isinstance(voltage, VoltageUnit):
            return cast(PowerUnit, Units.derive(
                PowerUnit.combine(voltage, self)
            ).named(name).symbol(symbol).make())
        raise TypeError("first parameter must be a VoltageUnit")

    def of(self, magnitude: float) -> Current:  # type: ignore
        return Current(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(  # type: ignore
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

    @overload
    def per(self, other: TimeUnit) -> VelocityUnit[CurrentUnit]: ...

    @overload
    def per[O: Unit](self, other: O) -> PerUnit[Self, O]: ...

    def per(self, other):  # type: ignore
        if isinstance(other, TimeUnit):
            return VelocityUnit.combine(self, other)
        return super().per(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[CurrentUnit, U]:
        return PerUnit.combine(self, other)

    def convert_from(self, magnitude: float, other_unit: CurrentUnit) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))


class EnergyUnit(Unit):
    def get_base_unit(self) -> EnergyUnit:
        return super().get_base_unit()

    @overload
    def per(self, other: TimeUnit) -> PowerUnit: ...  # type: ignore

    @overload
    def per[O: Unit](self, other: O) -> PerUnit[Self, O]: ...

    def per[U: Unit](  # type: ignore
        self, other: U
    ) -> PowerUnit | PerUnit[EnergyUnit, U]:
        if isinstance(other, TimeUnit):
            return PowerUnit.combine(self, other)
        if isinstance(other, Unit):
            return self.per_unit(other)
        self._raiseTypeErrorPer(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[EnergyUnit, U]:
        return PerUnit.combine(self, other)

    def convert_from(self, magnitude: float, other_unit: EnergyUnit) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))

    def of(self, magnitude: float) -> Energy:  # type: ignore
        return Energy(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(  # type: ignore
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
    def get_base_unit(self) -> TemperatureUnit:
        return super().get_base_unit()

    def of(self, magnitude: float) -> Temperature:  # type: ignore
        return Temperature(magnitude, self.to_base_units(magnitude), self)

    def of_base_units(  # type: ignore
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

    @overload
    def per(self, other: TimeUnit) -> VelocityUnit[TemperatureUnit]: ...

    @overload
    def per[O: Unit](self, other: O) -> PerUnit[Self, O]: ...

    def per(self, other):  # type: ignore
        if isinstance(other, TimeUnit):
            return VelocityUnit.combine(self, other)
        return super().per(other)

    def per_unit[U: Unit](self, other: U) -> PerUnit[TemperatureUnit, U]:
        return PerUnit.combine(self, other)

    def convert_from(
        self, magnitude: float, other_unit: TemperatureUnit
    ) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))


class PerUnit[A: Unit, B: Unit](Unit):
    def __init__(
        self,
        base_unit: Unit | PerUnit[A, B] | None,
        numerator: A,
        denominator: B,
    ) -> None:
        self._m_numerator = numerator
        self._m_denominator = denominator
        super().__init__(
            base_unit=base_unit,
            to_base_converter=UnaryFunction(lambda x: (
                x * (
                    numerator.to_base_units(1.0)
                    / denominator.to_base_units(1.0)))),
            from_base_converter=UnaryFunction(lambda x: (
                x * (
                    numerator.from_base_units(1.0)
                    * denominator.to_base_units(1.0)))),
            unit_name=numerator.name() + " per " + denominator.name(),
            unit_symbol=numerator.symbol() + "/" + denominator.symbol(),
        )
        self._base_unit = base_unit if base_unit is not None else self
        self._source: str = ""

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

    def of(self, magnitude: float) -> Per[A, B]:  # type: ignore
        try:
            return self.of_native(magnitude)
        finally:
            self._source = "of"

    def of_base_units(  # type: ignore
        self,
        base_unit_magnitude: float
    ) -> Per[A, B]:
        try:
            return self.of_native_base_units(base_unit_magnitude)
        finally:
            self._source = "of_base_units"

    def of_native(self, magnitude: float) -> Per[A, B]:
        self._source = "of_native"
        return Per(magnitude, self.to_base_units(magnitude), self)

    def of_native_base_units(
        self, base_unit_magnitude: float
    ) -> Per[A, B]:
        self._source = "of_native_base_units"
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
            base_unit=base_unit,
            to_base_converter=UnaryFunction(lambda x: (
                x * a.to_base_units(1.0) * b.to_base_units(1.0))),
            from_base_converter=UnaryFunction(lambda x: (
                x * a.from_base_units(1.0) * b.from_base_units(1.0))),
            unit_name=a.name() + "-" + b.name(),
            unit_symbol=a.symbol() + "*" + b.symbol(),
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

    def of(self, magnitude: float) -> Mult[A, B]:  # type: ignore
        return self.of_native(magnitude)

    def of_base_units(  # type: ignore
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

    def convert_from(
        self, magnitude: float, other_unit: Unit
    ) -> float:
        return self.from_base_units(other_unit.to_base_units(magnitude))


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
                "Numerator must be dimensionless"
                f"in a FrequencyUnit, got {type(numerator)}"
            )
        if not isinstance(denominator, TimeUnit):
            raise TypeError(
                "Denominator must be a time unit"
                f"in a FrequencyUnit, got {type(denominator)}"
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

    @overload
    def per(self, other: TimeUnit) -> VelocityUnit[FrequencyUnit]: ...

    @overload
    def per[O: Unit](self, other: O) -> PerUnit[Self, O]: ...

    def per(self, other):  # type: ignore
        if isinstance(other, TimeUnit):
            return VelocityUnit.combine(self, other)
        return super().per(other)


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
                "Numerator must be a distance unit in a LinearVelocityUnit, "
                f"got {type(numerator)}"
            )
        if not isinstance(denominator, TimeUnit):
            raise TypeError(
                "Denominator must be a time unit in a LinearVelocityUnit, "
                f"got {type(denominator)}"
            )
        return LinearVelocityCache.combine(numerator, denominator)

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

    @overload
    def per(self, other: TimeUnit) -> LinearAccelerationUnit[Self]: ...

    @overload
    def per[O: Unit](self, other: O) -> PerUnit[Self, O]: ...

    def per(self, other):  # type: ignore
        if isinstance(other, TimeUnit):
            return LinearAccelerationUnit.combine(self, other)
        return super().per(other)


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
                "Numerator must be dimensionless in a AngularVelocityUnit, "
                f"got {type(numerator)}"
            )
        if not isinstance(denominator, TimeUnit):
            raise TypeError(
                "Denominator must be a time unit in a AngularVelocityUnit, "
                f"got {type(denominator)}"
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

    @overload
    def per(self, other: TimeUnit) -> AngularAccelerationUnit[Self]: ...

    @overload
    def per[O: Unit](self, other: O) -> PerUnit[Self, O]: ...

    def per(self, other):  # type: ignore
        if isinstance(other, TimeUnit):
            return AngularAccelerationUnit.combine(self, other)
        return super().per(other)


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

    def per[O: Unit](self, other: O) -> PerUnit[Self, O]:  # type: ignore
        return super().per(other)

    def get_unit(self) -> U:
        return self.numerator()

    def get_period(self) -> TimeUnit:
        return self.denominator()


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
                "Numerator must be an angular velocity unit in a "
                f"in a AngularAccelerationUnit, got {type(numerator)}"
            )
        if not isinstance(denominator, TimeUnit):
            raise TypeError(
                f"Denominator must be a time unit "
                f"in a AngularAccelerationUnit, got {type(denominator)}"
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

    def per[O: Unit](self, other: O) -> PerUnit[Self, O]:  # type: ignore
        return super().per(other)


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
    def combine[O: Unit](
        numerator: O, denominator: TimeUnit
    ) -> VelocityUnit[O]:
        if not isinstance(denominator, TimeUnit):
            raise TypeError(
                "Denominator must be a time unit in a VelocityUnit, "
                f"got {type(denominator)}"
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

    def per[O: Unit](self, other: O) -> PerUnit[Self, O]:  # type: ignore
        return super().per(other)


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
                "Numerator (a) must be a mass unit in a ForceUnit, "
                f"got {type(a)}"
            )
        if not isinstance(b, LinearAccelerationUnit):
            raise TypeError(
                "Denominator (b) must be a linear acceleration unit "
                f"in a ForceUnit, got {type(b)}"
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

    def per[O: Unit](self, other: O) -> PerUnit[Self, O]:  # type: ignore
        return super().per(other)


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
                f"Numerator (a) must be a distance unit in a TorqueUnit, "
                f"got {type(a)}"
            )
        if not isinstance(b, ForceUnit):
            raise TypeError(
                "Denominator (b) must be a force unit "
                f"in a TorqueUnit, got {type(b)}"
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

    def per[O: Unit](self, other: O) -> PerUnit[Self, O]:  # type: ignore
        return super().per(other)


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
                "Numerator (a) must be a distance unit "
                f"in a LinearMomentumUnit, got {type(a)}"
            )
        if not isinstance(b, LinearVelocityUnit):
            raise TypeError(
                "Denominator (b) must be a force unit "
                f"in a LinearMomentumUnit, got {type(b)}"
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

    def per[O: Unit](self, other: O) -> PerUnit[Self, O]:  # type: ignore
        return super().per(other)


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
                f"in an AngularMomentumUnit, got {type(a)}"
            )
        if not isinstance(b, DistanceUnit):
            raise TypeError(
                "Denominator (b) must be a distance unit "
                f"in an AngularMomentumUnit, got {type(b)}"
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

    @overload
    def per[O: AngularVelocityUnit](
        self, other: O
    ) -> MomentOfInertiaUnit[Self, O]: ...

    @overload
    def per(self, other: TimeUnit) -> VelocityUnit[Self]: ...

    @overload
    def per[O: Unit](self, other: O) -> PerUnit[Self, O]: ...

    # type: ignore[override]
    def per(self, other):  # type: ignore
        # type: ignore[override]
        if isinstance(other, AngularVelocityUnit):
            return MomentOfInertiaUnit.combine(self, other)
        elif isinstance(other, TimeUnit):
            return VelocityUnit.combine(self, other)
        return super().per(other)


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
                f"in a MomentOfInteriaUnit, got {type(numerator)}"
            )
        if not isinstance(denominator, AngularVelocityUnit):
            raise TypeError(
                "Denominator must be an angular velocity unit "
                f"in a MomentOfInteriaUnit, got {type(denominator)}"
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

    def per[O: Unit](self, other: O) -> PerUnit[Self, O]:  # type: ignore
        return super().per(other)


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
](MultUnit[A, B]):
    def __init__(self, energy: A, time: B) -> None:
        base = (
            None
            if energy.is_base_unit() and time.is_base_unit()
            else PowerUnit.combine(
                energy.get_base_unit(), time.get_base_unit()  # type: ignore
            )
        )
        super().__init__(base, energy, time)  # type: ignore

    @overload  # type: ignore
    @staticmethod
    def combine[A_: EnergyUnit, B_: TimeUnit](
        a: A_, b: B_
    ) -> PowerUnit[A_, B_]: ...

    @overload
    @staticmethod
    def combine[A_: VoltageUnit, B_: CurrentUnit](
        a: A_, b: B_
    ) -> PowerUnit[A_, B_]: ...

    @overload
    @staticmethod
    def combine[A_: CurrentUnit, B_: VoltageUnit](
        a: A_, b: B_
    ) -> PowerUnit[A_, B_]: ...

    @overload
    @staticmethod
    def combine[A_: AngularVelocityUnit, B_: TorqueUnit](
        a: A_, b: B_
    ) -> PowerUnit[A_, B_]: ...

    @overload
    @staticmethod
    def combine[A_: TorqueUnit, B_: AngularVelocityUnit](
        a: A_, b: B_
    ) -> PowerUnit[A_, B_]: ...

    @staticmethod
    def combine(a, b):
        ALLOWED_PAIRS: tuple[  # type: ignore[annotation-unchecked]
            tuple[type[Unit], type[Unit]],
            ...
        ] = (
            (EnergyUnit, TimeUnit),
            (VoltageUnit, CurrentUnit),
            (CurrentUnit, VoltageUnit),
            (AngularVelocityUnit, TorqueUnit),
            (TorqueUnit, AngularVelocityUnit)
        )
        try:
            type_i = next(
                (i for i, (t, _) in enumerate(ALLOWED_PAIRS)
                 if t == type(a))
            )
            expect_denom_type = ALLOWED_PAIRS[type_i][1]
            if not isinstance(b, expect_denom_type):
                raise TypeError(
                    "Second element does not have correct matching type: "
                    f"expected {expect_denom_type}, but "
                    f"got invalid {type(b)}")
            return PowerCache.combine(a, b)
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

    def per[O: Unit](self, other: O) -> PerUnit[Self, O]:  # type: ignore
        return super().per(other)


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
                "Numerator must be a voltage unit in a ResistanceUnit, "
                f"got {type(numerator)}"
            )
        if not isinstance(denominator, CurrentUnit):
            raise TypeError(
                "Denominator must be a current unit "
                f"in a ResistanceUnit, got {type(denominator)}"
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

    def per[O: Unit](self, other: O) -> PerUnit[Self, O]:  # type: ignore
        return super().per(other)


ResistanceCache: CombinatoryUnitCache[
    VoltageUnit, CurrentUnit, ResistanceUnit
] = CombinatoryUnitCache(ResistanceUnit)


class AccelerationUnit[U: Unit](PerUnit[VelocityUnit[U], TimeUnit]):
    @overload
    def __init__(self, velocity_unit: U, period: TimeUnit): ...  # type: ignore

    @overload
    def __init__(
        self,
        base_unit: AccelerationUnit[U],
        to_base_converter: UnaryFunction,
        from_base_converter: UnaryFunction,
        name: str,
        symbol: str
    ):
        ...

    def __init__(  # type: ignore
        self,
        vu_bu,
        tbc_p,
        fbc=None,
        name=None,
        symbol=None
    ):
        if isinstance(vu_bu, Unit) and isinstance(tbc_p, TimeUnit):
            base = (
                None
                if vu_bu.is_base_unit() and tbc_p.is_base_unit()
                else AccelerationUnit.combine(
                    vu_bu.get_base_unit(),  # type: ignore
                    tbc_p.get_base_unit()
                )
            )
            super().__init__(base, vu_bu, tbc_p)  # type: ignore
        elif (
            isinstance(vu_bu, AccelerationUnit)
            and isinstance(tbc_p, UnaryFunction)
            and isinstance(fbc, UnaryFunction)
            and name is not None
            and symbol is not None
        ):
            super().__init__(  # type: ignore # noqa: S930
                base_unit=vu_bu,
                to_base_converter=tbc_p,   # type: ignore # noqa: S930
                from_base_converter=fbc,   # type: ignore # noqa: S930
                unit_name=name,   # type: ignore # noqa: S930
                unit_symbol=symbol   # type: ignore # noqa: S930
            )
        raise NotImplementedError(
            "AccelerationUnit constructor takes (Unit, TimeUnit) or "
            "(AccelerationUnit, UnaryFunction, UnaryFunction, str, str)"
        )

    @staticmethod
    def combine[O: Unit](  # type: ignore
        numerator: VelocityUnit[O],
        denominator: TimeUnit
    ) -> AccelerationUnit[VelocityUnit[O]]:
        if not isinstance(numerator, VelocityUnit):
            raise TypeError(
                "Numerator must be a voltage unit in a ResistanceUnit, "
                f"got {type(numerator)}"
            )
        if not isinstance(denominator, TimeUnit):
            raise TypeError(
                "Denominator must be a current unit "
                f"in a ResistanceUnit, got {type(denominator)}"
            )
        return AccelerationUnitCache.combine(
            numerator, denominator)  # type: ignore

    def get_base_unit(self) -> AccelerationUnit:
        return cast(AccelerationUnit, super().get_base_unit())

    def of(  # type: ignore
        self, magnitude: float
    ) -> AccelerationUnit:
        return self.__init__(magnitude,  # type: ignore
                             self.to_base_units(magnitude),
                             self)

    def of_base_units(  # type: ignore[override]
        self, base_unit_magnitude: float
    ) -> Acceleration:
        return Acceleration(
            self.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            self
        )

    @overload  # type: ignore[no-overload-impl]
    def per(  # type: ignore
        self, other: "TimeUnit"  # noqa: UP037
    ) -> VelocityUnit[AccelerationUnit[U]]: ...

    @overload
    def per[O: "Unit"](
        self, other: O
    ) -> PerUnit[Self, O]: ...

    def per(self, other):  # type: ignore
        return super().per(other)


AccelerationUnitCache: CombinatoryUnitCache[
    VelocityUnit[Unit], TimeUnit, AccelerationUnit
] = CombinatoryUnitCache(AccelerationUnit)


@final
class BaseUnits:
    BaseDistanceUnit: Final = DistanceUnit(  # noqa: S116
        base_unit=None, base_unit_equivalent=1.0,
        unit_name="Meter", unit_symbol="m")
    BaseTimeUnit: Final = TimeUnit(  # noqa: S116
        base_unit=None, base_unit_equivalent=1.0,
        unit_name="Second", unit_symbol="s")
    BaseMassUnit: Final = MassUnit(  # noqa: S116
        base_unit=None, base_unit_equivalent=1.0,
        unit_name="Kilogram", unit_symbol="Kg")
    BaseAngleUnit: Final = AngleUnit(  # noqa: S116
        base_unit=None, base_unit_equivalent=1.0,
        unit_name="Radian", unit_symbol="rad")
    BaseVoltageUnit: Final = VoltageUnit(  # noqa: S116
        base_unit=None, base_unit_equivalent=1.0,
        unit_name="Volt", unit_symbol="V")
    BaseCurrentUnit: Final = CurrentUnit(  # noqa: S116
        base_unit=None, base_unit_equivalent=1.0,
        unit_name="Amp", unit_symbol="A")
    BaseEnergyUnit: Final = EnergyUnit(  # noqa: S116
        base_unit=None, base_unit_equivalent=1.0,
        unit_name="Joule", unit_symbol="J")
    BaseTemperatureUnit: Final = TemperatureUnit(  # noqa: S116
        base_unit=None,
        to_base_converter=UnaryFunction.IDENTITY,
        from_base_converter=UnaryFunction.IDENTITY,
        unit_name="Kelvin", unit_symbol="K"
    )
    BaseValue: Final = DimensionlessUnit(  # noqa: S116
        base_unit=None, base_unit_equivalent=1.0,
        unit_name="<?>", unit_symbol="<?>")


@final
class Units:
    Value: Final = BaseUnits.BaseValue  # type: ignore # noqa: S116
    Percent: Final = (UnitBuilder(Value)  # type: ignore # noqa: S116
                      .split_into(100.0)
                      .named("Percent").symbol("%")
                      .make_and_cast(DimensionlessUnit))

    Meters: Final = BaseUnits.BaseDistanceUnit  # type: ignore # noqa: S116
    Meter: Final = Meters  # type: ignore # noqa: S116
    Millimeters: Final = (UnitBuilder(Meters)  # type: ignore # noqa: S116
                          .split_into(1000.0)
                          .named("Millimeter").symbol("mm")
                          .make_and_cast(DistanceUnit))
    Millimeter: Final = Millimeters  # type: ignore # noqa: S116
    Centimeters: Final = (UnitBuilder(Meters)  # type: ignore # noqa: S116
                          .split_into(100.0)
                          .named("Centimeter").symbol("cm")
                          .make_and_cast(DistanceUnit))
    Centimeter: Final = Centimeters  # type: ignore # noqa: S116
    Inches: Final = (UnitBuilder(Millimeters)  # type: ignore # noqa: S116
                     .aggregate(25.4)
                     .named("Inch").symbol("in")
                     .make_and_cast(DistanceUnit))
    Inch: Final = Inches  # type: ignore # noqa: S116
    Feet: Final = (UnitBuilder(Inches)  # type: ignore # noqa: S116
                   .aggregate(12.0)
                   .named("Foot").symbol("ft")
                   .make_and_cast(DistanceUnit))
    Foot: Final = Feet  # type: ignore # noqa: S116

    Seconds: Final = BaseUnits.BaseTimeUnit  # type: ignore # noqa: S116
    Second: Final = Seconds  # type: ignore # noqa: S116
    Milliseconds: Final = (UnitBuilder(Seconds)  # type: ignore # noqa: S116
                           .split_into(1000.0).named("Millisecond")
                           .symbol("ms")
                           .make_and_cast(TimeUnit))
    Millisecond: Final = Milliseconds  # type: ignore # noqa: S116
    Microseconds: Final = (UnitBuilder(Seconds)  # type: ignore # noqa: S116
                           .split_into(1_000_000.0).named("Microsecond")
                           .symbol("us")
                           .make_and_cast(TimeUnit))
    Microsecond: Final = Microseconds  # type: ignore # noqa: S116
    Nanoseconds: Final = (UnitBuilder(Seconds)  # type: ignore # noqa: S116
                          .split_into(1_000_000_000.0).named("Nanosecond")
                          .symbol("ns")
                          .make_and_cast(TimeUnit))
    Nanosecond: Final = Nanoseconds  # type: ignore # noqa: S116
    Minutes: Final = (UnitBuilder(Seconds)  # type: ignore # noqa: S116
                      .aggregate(60.0).named("Minute")
                      .symbol("min")
                      .make_and_cast(TimeUnit))
    Minute: Final = Minutes  # type: ignore # noqa: S116
    Hours: Final = (UnitBuilder(Minutes)  # type: ignore # noqa: S116
                    .aggregate(60.0).named("Hour")
                    .symbol("hr")
                    .make_and_cast(TimeUnit))
    Hour: Final = Hours  # type: ignore # noqa: S116

    Radians: Final = BaseUnits.BaseAngleUnit  # type: ignore # noqa: S116
    Radian: Final = Radians  # type: ignore # noqa: S116
    Revolutions: Final = (UnitBuilder(Radians)  # type: ignore # noqa: S116
                          .aggregate(2.0 * math.pi)
                          .named("Revolution").symbol("R")
                          .make_and_cast(AngleUnit))
    Revolution: Final = Revolutions  # type: ignore # noqa: S116
    Rotations: Final = Revolutions  # type: ignore # noqa: S116
    Rotation: Final = Rotations  # type: ignore # noqa: S116
    Degrees: Final = (UnitBuilder(Revolutions)  # type: ignore # noqa: S116
                      .split_into(360.0)
                      .named("Degree").symbol("\xb0")
                      .make_and_cast(AngleUnit))
    Degree: Final = Degrees  # type: ignore # noqa: S116

    MetersPerSecond: Final = Meters.per(Second)  # type: ignore # noqa: S116
    FeetPerSecond: Final = Feet.per(Second)  # type: ignore # noqa: S116
    InchesPerSecond: Final = Inches.per(Second)  # type: ignore # noqa: S116
    RevolutionsPerSecond: Final = (  # type: ignore # noqa: S116
        Revolutions.per(Second))
    RotationsPerSecond: Final = (  # type: ignore # noqa: S116
        Rotations.per(Second))
    RotationsPerMinute: Final = (  # type: ignore # noqa: S116
        Rotations.per(Minute))
    RPM: Final = RotationsPerMinute  # type: ignore # noqa: S116
    RadiansPerSecond: Final = Radians.per(Second)  # type: ignore # noqa: S116
    DegreesPerSecond: Final = Degrees.per(Second)  # type: ignore # noqa: S116

    Hertz: Final = (UnitBuilder(Value.per(Second))  # type: ignore # noqa: S116
                    .named("Hertz").symbol("Hz")
                    .make_and_cast(FrequencyUnit[DimensionlessUnit]))
    Millihertz: Final = (UnitBuilder(Hertz)  # type: ignore # noqa: S116
                         .split_into(1000.0)
                         .named("Millihertz").symbol("MHz")
                         .make_and_cast(FrequencyUnit[DimensionlessUnit]))

    MetersPerSecondPerSecond: Final = (  # type: ignore # noqa: S116
        MetersPerSecond.per(Second))
    FeetPerSecondPerSecond: Final = (  # type: ignore # noqa: S116
        FeetPerSecond.per(Second))
    InchesPerSecondPerSecond: Final = (  # type: ignore # noqa: S116
        InchesPerSecond.per(Second))
    RotationsPerSecondPerSecond: Final = (  # type: ignore # noqa: S116
        RotationsPerSecond.per(Second))
    RadiansPerSecondPerSecond: Final = (  # type: ignore # noqa: S116
        RadiansPerSecond.per(Second))
    DegreesPerSecondPerSecond: Final = (  # type: ignore # noqa: S116
        DegreesPerSecond.per(Second))
    Gs: Final = (  # type: ignore # noqa: S116
        UnitBuilder(MetersPerSecondPerSecond)
        .aggregate(9.80665)
        .named("G").symbol("G")
        .make_and_cast(LinearAccelerationUnit[
            LinearVelocityUnit[DistanceUnit]
        ])
    )

    Kilograms: Final = BaseUnits.BaseMassUnit  # type: ignore # noqa: S116
    Kilogram: Final = Kilograms  # type: ignore # noqa: S116
    Grams: Final = (UnitBuilder(Kilograms)  # type: ignore # noqa: S116
                    .split_into(1000.0)
                    .named("Gram").symbol("g")
                    .make_and_cast(MassUnit))
    Gram: Final = Grams  # type: ignore # noqa: S116
    Pounds: Final = (UnitBuilder(Grams)  # type: ignore # noqa: S116
                     .aggregate(453.592)
                     .named("Pound").symbol("lb.")
                     .make_and_cast(MassUnit))
    Pound: Final = Pounds  # type: ignore # noqa: S116
    Ounces: Final = (UnitBuilder(Pounds)  # type: ignore # noqa: S116
                     .split_into(16.0)
                     .named("Ounce").symbol("oz.")
                     .make_and_cast(MassUnit))
    Ounce: Final = Ounces  # type: ignore # noqa: S116

    Newtons: Final = (  # type: ignore # noqa: S116
        UnitBuilder(Kilograms.mult(MetersPerSecondPerSecond))
        .named("Newton").symbol("N")
        .make_and_cast(ForceUnit[
                            MassUnit,
                            LinearAccelerationUnit[
                                LinearVelocityUnit[DistanceUnit]
                            ]
                        ]))
    Newton: Final = Newtons  # type: ignore # noqa: S116
    PoundsForce: Final = (  # type: ignore # noqa: S116
        UnitBuilder(Pounds.mult(Gs))
        .named("Pound-force").symbol("lbsf.")
        .make_and_cast(ForceUnit[
            MassUnit,
            LinearAccelerationUnit[
                LinearVelocityUnit[DistanceUnit]
            ]
        ])
    )
    PoundForce: Final = PoundsForce  # type: ignore # noqa: S116
    OuncesForce: Final = (  # type: ignore # noqa: S116
        UnitBuilder(Ounces.mult(Gs))
        .named("Ounce-force").symbol("ozf")
        .make_and_cast(ForceUnit[
            MassUnit,
            LinearAccelerationUnit[
                LinearVelocityUnit[DistanceUnit]
            ]
        ])
    )
    OunceForce: Final = OuncesForce  # type: ignore # noqa: S116

    NewtonMeters: Final = (  # type: ignore # noqa: S116
        Meters.mult_as_torque(Newtons))
    NewtonMeter: Final = NewtonMeters  # type: ignore # noqa: S116
    PoundFeet: Final = (  # type: ignore # noqa: S116
        Feet.mult_as_torque(PoundsForce))
    PoundFoot: Final = PoundFeet  # type: ignore # noqa: S116
    PoundInches: Final = (  # type: ignore # noqa: S116
        Inches.mult_as_torque(PoundsForce))
    PoundInch: Final = PoundInches  # type: ignore # noqa: S116
    OunceInches: Final = (  # type: ignore # noqa: S116
        Inches.mult_as_torque(OuncesForce))
    OunceInch: Final = OunceInches  # type: ignore # noqa: S116

    KilogramMetersPerSecond: Final = (  # type: ignore # noqa: S116
        Kilograms.mult(MetersPerSecond).cast(MultUnit[
            MassUnit,
            LinearVelocityUnit[DistanceUnit]
        ])
    )
    PoundInchesPerSecond: Final = (  # type: ignore # noqa: S116
        Pounds.mult(InchesPerSecond).cast(MultUnit[
            MassUnit,
            LinearVelocityUnit[DistanceUnit]
        ])
    )

    KilogramMetersSquaredPerSecond: Final = (  # type: ignore # noqa: S116
        KilogramMetersPerSecond.mult(Meters).cast(MultUnit[
            MassUnit,
            MultUnit[
                MassUnit,
                LinearVelocityUnit[DistanceUnit]
            ]
        ])
    )
    PoundInchesSquaredPerSecond: Final = (  # type: ignore # noqa: S116
        PoundInchesPerSecond.mult(Inches).cast(MultUnit[
            MassUnit,
            MultUnit[
                MassUnit,
                LinearVelocityUnit[DistanceUnit]
            ]
        ])
    )

    KilogramSquareMeters: Final = (  # type: ignore # noqa: S116
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
    PoundSquareInches: Final = (  # type: ignore # noqa: S116
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

    Volts: Final = BaseUnits.BaseVoltageUnit  # type: ignore # noqa: S116
    Volt: Final = Volts  # type: ignore # noqa: S116
    Millivolts: Final = (UnitBuilder(Volts)  # type: ignore # noqa: S116
                         .split_into(1000.0)
                         .named("Millivolt").symbol("mV")
                         .make_and_cast(VoltageUnit))
    Millivolt: Final = Millivolts  # type: ignore # noqa: S116

    Amps: Final = BaseUnits.BaseCurrentUnit  # type: ignore # noqa: S116
    Amp: Final = Amps  # type: ignore # noqa: S116
    Milliamps: Final = (UnitBuilder(Amps)  # type: ignore # noqa: S116
                        .split_into(1000.0)
                        .named("Milliamp").symbol("mA")
                        .make_and_cast(CurrentUnit))
    Milliamp: Final = Milliamps  # type: ignore # noqa: S116

    Ohms: Final = (UnitBuilder(Volts.per(Amp))  # type: ignore # noqa: S116
                   .named("Ohm").symbol("\u03a9")
                   .make_and_cast(ResistanceUnit))
    Ohm: Final = Ohms  # type: ignore # noqa: S116
    KiloOhms: Final = (UnitBuilder(Ohms)  # type: ignore # noqa: S116
                       .aggregate(1000.0)
                       .named("KiloOhm").symbol("k\u03a9")
                       .make_and_cast(ResistanceUnit))
    KiloOhm: Final = KiloOhms  # type: ignore # noqa: S116
    MilliOhms: Final = (UnitBuilder(Ohms)  # type: ignore # noqa: S116
                        .split_into(1000.0)
                        .named("MilliOhm").symbol("m\u03a9")
                        .make_and_cast(ResistanceUnit))
    MilliOhm: Final = MilliOhms  # type: ignore # noqa: S116

    Joules: Final = BaseUnits.BaseEnergyUnit  # type: ignore # noqa: S116
    Joule: Final = Joules  # type: ignore # noqa: S116
    Millijoules: Final = (UnitBuilder(Joules)  # type: ignore # noqa: S116
                          .split_into(1000.0)
                          .named("Millijoule").symbol("mJ")
                          .make_and_cast(EnergyUnit))
    Millijoule: Final = Millijoules  # type: ignore # noqa: S116
    Kilojoules: Final = (UnitBuilder(Joules)  # type: ignore # noqa: S116
                         .aggregate(1000.0)
                         .named("Kilojoule").symbol("kJ")
                         .make_and_cast(EnergyUnit))
    Kilojoule: Final = Kilojoules  # type: ignore # noqa: S116

    Watts: Final = (  # type: ignore # noqa: S116
        UnitBuilder(Joules.per(Second))
        .named("Watt").symbol("W")
        .make_and_cast(PerUnit[EnergyUnit, TimeUnit]))
    Watt: Final = Watts  # type: ignore # noqa: S116
    Milliwatts: Final = (UnitBuilder(Watts)  # type: ignore # noqa: S116
                         .split_into(1000.0)
                         .named("Milliwatt").symbol("mW")
                         .make_and_cast(PerUnit[EnergyUnit, TimeUnit]))
    Milliwatt: Final = Milliwatts  # type: ignore # noqa: S116
    Horsepower: Final = (UnitBuilder(Watts)  # type: ignore # noqa: S116
                         .aggregate(745.7)
                         .named("Horsepower").symbol("HP")
                         .make_and_cast(PerUnit[EnergyUnit, TimeUnit]))

    Kelvin: Final = BaseUnits.BaseTemperatureUnit  # type: ignore # noqa: S116
    Celsius: Final = (UnitBuilder(Kelvin)  # type: ignore # noqa: S116
                      .offset(273.15)
                      .named("Celsius").symbol("\xb0C")
                      .make_and_cast(TemperatureUnit))
    Fahrenheit: Final = (UnitBuilder(Celsius)  # type: ignore # noqa: S116
                         .mapping_input_range(0.0, 100.0)
                         .to_output_range(32.0, 212.0)
                         .named("Fahrenheit").symbol("\xb0F")
                         .make_and_cast(TemperatureUnit))

    VoltsPerMeterPerSecond: Final = (  # type: ignore # noqa: S116
        Volts.per(MetersPerSecond)
        .cast(PerUnit[
            VoltageUnit,
            LinearVelocityUnit[DistanceUnit]
        ]))

    VoltsPerMeterPerSecondSquared: Final = (  # type: ignore # noqa: S116
        Volts.per(MetersPerSecondPerSecond).cast(PerUnit[
            VoltageUnit,
            LinearAccelerationUnit[LinearVelocityUnit[DistanceUnit]]
        ])
    )
    VoltsPerRadianPerSecond: Final = (  # type: ignore # noqa: S116
        Volts.per(RadiansPerSecond)
        .cast(PerUnit[
            VoltageUnit,
            AngularVelocityUnit[AngleUnit]
        ]))
    VoltsPerRadianPerSecondSquared: Final = (  # type: ignore # noqa: S116
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
