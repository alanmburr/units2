from __future__ import annotations

import math
from decimal import Decimal
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Protocol,
    Self,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

if TYPE_CHECKING:
    from .units import (
        AccelerationUnit,  # noqa: F401, RUF100
        AngleUnit,  # noqa: F401, RUF100
        AngularAccelerationUnit,  # noqa: F401, RUF100
        AngularMomentumUnit,  # noqa: F401, RUF100
        AngularVelocityUnit,  # noqa: F401, RUF100
        CurrentUnit,  # noqa: F401, RUF100
        DimensionlessUnit,  # noqa: F401, RUF100
        DistanceUnit,  # noqa: F401, RUF100
        EnergyUnit,  # noqa: F401, RUF100
        ForceUnit,  # noqa: F401, RUF100
        FrequencyUnit,  # noqa: F401, RUF100
        LinearAccelerationUnit,  # noqa: F401, RUF100
        LinearMomentumUnit,  # noqa: F401, RUF100
        LinearVelocityUnit,  # noqa: F401, RUF100
        MassUnit,  # noqa: F401, RUF100
        MomentOfInertiaUnit,  # noqa: F401, RUF100
        MultUnit,  # noqa: F401, RUF100
        PerUnit,  # noqa: F401, RUF100
        PowerUnit,  # noqa: F401, RUF100
        ResistanceUnit,  # noqa: F401, RUF100
        TemperatureUnit,  # noqa: F401, RUF100
        TimeUnit,  # noqa: F401, RUF100
        TorqueUnit,  # noqa: F401, RUF100
        Unit,  # noqa: F401, RUF100
        VelocityUnit,  # noqa: F401, RUF100
        VoltageUnit,  # noqa: F401, RUF100
    )


A = TypeVar("A", bound="Unit")
B = TypeVar("B", bound="Unit")

EQUIVALENCE_THRESHOLD: float = 1e-12


def _double_compare(d1sf: float, d2sf: float) -> int:
    """
    Mimics Java's Double.compare behavior for Python floats.
    Returns:
        0  if d1 == d2
        -1 if d1 < d2
        1  if d1 > d2
    Handles NaN and signed zero cases like Java.
    """
    d1 = float(d1sf)
    d2 = float(d2sf)

    # Handle NaN cases
    if math.isnan(d1):
        return 1 if not math.isnan(d2) else 0
    if math.isnan(d2):
        return -1

    # Handle signed zero
    if d1 == d2:
        # Distinguish between +0.0 and -0.0
        if math.copysign(1.0, d1) != math.copysign(1.0, d2):
            return 1 if math.copysign(1.0, d1) > math.copysign(1.0, d2) else -1
        return 0

    return -1 if d1 < d2 else 1


@runtime_checkable
class Measure[U: "Unit"](Protocol):
    EQUIVALENCE_THRESHOLD: float = 1e-12

    def magnitude(self) -> float:
        m = getattr(self, '_magnitude', None)
        if m is not None:
            return m
        raise NotImplementedError("Measure.magnitude()")

    def base_unit_magnitude(self) -> float:
        m = getattr(self, '_base_unit_magnitude', None)
        if m is not None:
            return m
        raise NotImplementedError("Measure.base_unit_magnitude()")

    def unit(self) -> U:
        u = getattr(self, '_unit', None)
        if u is not None:
            return u
        raise NotImplementedError("Measure.unit()")

    def in_unit(self, unit: "Unit") -> float:  # noqa: UP037
        if self.unit() == unit:
            from .units import PerUnit as PU
            if (
                isinstance(self.unit(), PU)
                and cast(PU, self.unit())._source in ['of', 'of_base_units']
            ):
                raise TypeError(
                    "Per WPILibJ wpiunits, you can't put a measure into unit "
                    "if the measure and the unit are both PerUnit-s and the "
                    "measure was created with .of() or .of_base_units(). See: "
                    "https://github.com/wpilibsuite/allwpilib/blob/main/"
                    "wpiunits/src/test/java/org/wpilib/units/PerUnitTest.java"
                )
            return self.magnitude()
        else:
            return unit.from_base_units(self.base_unit_magnitude())

    def base_unit(self) -> "Unit":  # noqa: UP037
        return self.unit().get_base_unit()

    def abs(self, unit: "Unit") -> float:  # noqa: UP037
        return abs(self.in_unit(unit))

    def copy_sign(
        self,
        other: Measure["Unit"],  # noqa: UP037
        unit: "Unit"  # noqa: UP037
    ) -> float:
        return math.copysign(self.in_unit(unit), other.in_unit(unit))

    def to(self, unit: U) -> float:
        if self.unit() == unit:
            return self.magnitude()
        if (
            self.base_unit() == unit.get_base_unit()
            or str(type(self.base_unit())) == str(type(unit.get_base_unit()))
        ):
            return unit.from_base_units(self.base_unit_magnitude())
        raise TypeError(
            "unit must have compatible dimensions, cannot convert "
            f"{type(unit.get_base_unit())} to {type(self.base_unit())}"
        )

    def unary_minus(self) -> Measure[U]:
        return self.unit().of(0.0 - self.magnitude())

    def plus(self, other: Measure[U]) -> Measure[U]:
        if not isinstance(other.base_unit(), type(self.base_unit())):
            raise TypeError(
                "other must have compatible dimensions, cannot add "
                f"{type(other.base_unit())} to {type(self.base_unit())}"
            )
        return self.base_unit().of(
            self.base_unit_magnitude()
            + other.base_unit_magnitude()
        )

    def minus(self, other: Measure[U]) -> Measure[U]:
        if not isinstance(other.base_unit(), type(self.base_unit())):
            raise TypeError(
                "other must have compatible dimensions, cannot subtract "
                f"{type(other.base_unit())} from {type(self.base_unit())}"
            )
        return self.base_unit().of(
            self.base_unit_magnitude()
            - other.base_unit_magnitude()
        )

    @overload
    def times(self, multiplier: float) -> U: ...

    @overload
    def times(self, multiplier: "DimensionlessUnit") -> U: ...  # noqa: UP037

    @overload
    def times[O: (
        Acceleration | Angle | AngularAcceleration | AngularMomentum |
        AngularVelocity | Distance | Energy | Force | Frequency |
        LinearAcceleration | LinearMomentum | LinearVelocity | Mass |
        MomentOfInertia | Mult | Power | Per | Temperature | Time | Torque |
        Velocity | Voltage | Resistance
    )](self, muliplier: O) -> "Mult[U, O]": ...  # noqa: UP037

    @overload
    def times(self, multiplier: Time) -> U: ...

    def times[O: Measure](  # type: ignore[misc] # noqa: S3776
        self,
        multiplier: O | float
    ) -> Measure["Mult[U, O]"] | Measure[U]:  # type: ignore # noqa: UP037
        if isinstance(multiplier, (float, int)):
            return self.unit().of_base_units(
                    self.base_unit_magnitude() * multiplier)

        from .units import DimensionlessUnit as DU

        if isinstance(multiplier.unit(), DU):
            return self.times(multiplier.base_unit_magnitude())  # type: ignore
        elif isinstance(self.unit(), DU):
            return cast(Measure, multiplier.times(self.base_unit_magnitude()))
        elif isinstance(multiplier, Measure):
            if isinstance(self, Velocity) and isinstance(multiplier, Time):
                from .units import Unit as UU
                from .units import VelocityUnit as VU
                return (cast(UU, cast(VU, self.unit()).numerator())
                        .of_base_units(
                            self.base_unit_magnitude()
                            * multiplier.base_unit_magnitude()))
            if isinstance(multiplier, (
                Acceleration, Angle,
                AngularAcceleration, AngularMomentum, AngularVelocity, Current,
                Distance, Energy, Force, Frequency, LinearAcceleration,
                LinearMomentum, LinearVelocity, Mass, MomentOfInertia,
                Mult, Power, Per, Temperature, Torque,
                Velocity, Voltage, Resistance
            )):
                from .units import MultUnit as MU
                return cast(
                    Measure,
                    MU.combine(self.unit(), multiplier.unit())
                    .of_base_units(
                        self.base_unit_magnitude()
                        * multiplier.base_unit_magnitude()
                    )
                )
            from .units import PerUnit as PU
            from .units import Unit as UU
            from .units import Units as US

            base_unit_result: Final = (
                self.base_unit_magnitude()
                * multiplier.base_unit_magnitude()
            )

            ret: Any = None
            if (
                isinstance(self.unit(), PU)
                and (
                    multiplier.base_unit() == (
                        cast(PU[UU, UU], multiplier)
                        .denominator()
                        .get_base_unit())
                )
            ) or (
                isinstance(self.base_unit(), PU)
                and (
                    multiplier.base_unit() == (
                        cast(PU[UU, UU], multiplier)
                        .denominator()
                        .get_base_unit())
                )
            ):
                ret = (cast(PU[UU, UU], multiplier)
                       .numerator().of_base_units(base_unit_result))
            elif (
                isinstance(self.unit(), PU)
                and isinstance(multiplier.unit(), PU)
                and (
                    (cast(PU[UU, UU], self)
                     .denominator().get_base_unit()) ==
                    (cast(PU[UU, UU], multiplier)
                     .numerator().get_base_unit()))
                and (
                    (cast(PU[UU, UU], self)
                     .numerator().get_base_unit()) ==
                    (cast(PU[UU, UU], multiplier)
                     .denominator().get_base_unit())
                )
            ):
                ret = US.Value.of(base_unit_result)
            if ret is not None:
                return cast(Measure, ret)
        raise TypeError(f"Incompatible type for times: {type(multiplier)}")

    def times_conversion_factor[O: "Unit"](
        self,
        factor: Measure["PerUnit[O, Unit]"]  # noqa: E501, UP037, RUF100
    ) -> Measure[O]:
        from .units import PerUnit as PU
        if not isinstance(factor, (Measure, PU)):
            raise TypeError(
                "factor must be a Measure or PerUnit "
                "for this times_conversion_factor"
                f", got {factor} ({type(factor)})"
            )
        return cast(
            Measure[O],
            factor.unit().get_base_unit().numerator().of_base_units(
                self.base_unit_magnitude() * factor.base_unit_magnitude())
        )

    def times_inverse(
        self,
        multiplier: Measure[
            "PerUnit[DimensionlessUnit, Unit]"  # noqa: E501, UP037, RUF100
        ]
    ) -> Dimensionless:
        from .units import Units
        return Units.Value.of_base_units(
            self.base_unit_magnitude() * multiplier.base_unit_magnitude()
        )

    def times_ratio[O: "Unit"](
        self,
        ratio: Measure["PerUnit[O, Unit]"]  # noqa: E501, UP037, RUF100
    ) -> Measure[O]:
        return cast(
            Measure[O],
            ImmutableMeasure.of_base_units(
                self.base_unit_magnitude() * ratio.base_unit_magnitude(),
                ratio.unit().numerator()
            )
        )

    @overload
    def div(self, multiplier: float | Dimensionless) -> Self: ...

    @overload
    def div[O: (
        Acceleration | Angle | AngularAcceleration | AngularMomentum |
        AngularVelocity | Distance | Energy | Force | Frequency |
        LinearAcceleration | LinearMomentum | LinearVelocity | Mass |
        MomentOfInertia | Mult | Power | Per | Temperature | Time | Torque |
        Velocity | Voltage | Resistance
    )](self, muliplier: O) -> "Per[Self, O]": ...  # noqa: UP037

    def div(self, divisor):  # type: ignore
        if isinstance(divisor, (float, int)):
            return self.unit().of_base_units(
                self.base_unit_magnitude() / divisor)

        from .units import PerUnit as PU

        if isinstance(divisor, Measure):
            from .units import DimensionlessUnit as DU
            from .units import TimeUnit as TU
            from .units import Unit as UU
            from .units import Units as US
            from .units import VelocityUnit as VU

            base_unit_result: Final = (  # type: ignore[annotation-unchecked]
                self.base_unit_magnitude()
                / divisor.base_unit_magnitude()
            )

            ret: Any = None  # type: ignore[annotation-unchecked]
            ratio = cast(PU[UU, UU], divisor.unit())
            if self.unit().get_base_unit() == divisor.unit().get_base_unit():
                ret = cast(Measure, US.Value.of_base_units(base_unit_result))
            elif isinstance(divisor, DU):
                ret = self.unit().of_base_units(base_unit_result)
            elif (
                isinstance(self.unit(), DU)
                and isinstance(divisor.unit(), PU)
            ):
                ret = (ratio.reciprocal()
                       .of_base_units(base_unit_result))
            # elif (
            #     isinstance(divisor.unit(), PU)
            #     and ratio.numerator().get_base_unit() == self.base_unit()
            # ):
            #     # this is faulty
            #     ret = (ratio.denominator()
            #            .of_base_units(self.base_unit_magnitude()))
            elif isinstance(divisor.unit(), TU):
                ret = (VU.combine(self.unit(), cast(TU, divisor.unit()))
                       .of_base_units(base_unit_result))

            if ret is not None:
                return cast(Measure, ret)
            else:
                my_unit = self.unit()
                their_unit = divisor.unit()
                new_unit = my_unit.per(their_unit)
                return new_unit.of(
                    self.magnitude() / divisor.magnitude()
                )
        elif isinstance(divisor, (
            Acceleration, Angle,
            AngularAcceleration, AngularMomentum, AngularVelocity, Current,
            Distance, Energy, Force, Frequency, LinearAcceleration,
            LinearMomentum, LinearVelocity, Mass, MomentOfInertia,
            Mult, Power, Per, Temperature, Time, Torque,
            Velocity, Voltage, Resistance
        )):
            return cast(
                Measure,
                PU.combine(self.unit(), divisor.unit())
                .of_base_units(
                    self.base_unit_magnitude()
                    / divisor.base_unit_magnitude()
                )
            )
        raise TypeError(f"Incompatible type for div: {type(divisor)}")

    def per[O: (
        AccelerationUnit | AngleUnit |
        AngularAccelerationUnit | AngularMomentumUnit |
        AngularVelocityUnit | CurrentUnit | DistanceUnit | EnergyUnit |
        ForceUnit | FrequencyUnit | LinearAccelerationUnit |
        LinearMomentumUnit | LinearVelocityUnit | MassUnit |
        MomentOfInertiaUnit | MultUnit | PowerUnit | PerUnit |
        TemperatureUnit | TimeUnit | TorqueUnit | VelocityUnit |
        VoltageUnit | ResistanceUnit | DimensionlessUnit
    )](
        self,
        divisor: O
    ) -> Measure["PerUnit[U, O]"]:  # noqa: UP037
        from .units import Unit as UU

        if isinstance(divisor, UU):
            return self.div(divisor.one())  # type: ignore

        raise TypeError(f"Incompatible type for per: {type(divisor)}")

    def divide_ratio[O: "Unit"](
        self,
        divisor: Measure["PerUnit[Unit, O]"]  # noqa: UP037
    ) -> Measure[O]:
        return ImmutableMeasure.of_base_units(
            (
                self.base_unit_magnitude()
                / divisor.base_unit_magnitude()
            ),
            divisor.unit().denominator()
        )

    def is_near(
        self,
        other: Self,
        variance_threshold_or_tolerance: float | Self
    ) -> bool:
        if isinstance(variance_threshold_or_tolerance, (float, int)):
            if (
                not self.unit().get_base_unit().equivalent(
                    other.unit().get_base_unit()
                )
            ):
                return False

            tolerance = abs(
                other.base_unit_magnitude()
                * variance_threshold_or_tolerance
            )
            return abs(
                self.base_unit_magnitude()
                - other.base_unit_magnitude()
            ) <= tolerance

        return abs(
            self.base_unit_magnitude() - other.base_unit_magnitude()
        ) <= abs(variance_threshold_or_tolerance.base_unit_magnitude())

    def is_equivalent(self, other: Measure[Unit]) -> bool:
        return (
            self.unit().get_base_unit() == other.unit().get_base_unit()
            and abs(
                self.base_unit_magnitude() - other.base_unit_magnitude()
            ) <= EQUIVALENCE_THRESHOLD
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, (float, int)):
            other = self.unit().of(other)
        if not isinstance(other, Measure):
            return False
        return self.is_equivalent(other)

    def compare_to(self, other: Measure["Unit"]) -> int:  # noqa: UP037
        return _double_compare(
            self.base_unit_magnitude(),
            other.base_unit_magnitude()
        )

    def gt(self, other: Measure["Unit"]) -> bool:  # noqa: UP037
        return self.compare_to(other) > 0

    def __gt__(self, other):
        if isinstance(other, (float, int)):
            other = self.unit().of(other)
        if not isinstance(other, type(self)):
            return False
        return self.gt(other)

    def gte(self, other: Measure["Unit"]) -> bool:  # noqa: UP037
        return self.compare_to(other) > 0 or self.is_equivalent(other)

    def __ge__(self, other):
        if isinstance(other, (float, int)):
            other = self.unit().of(other)
        if not isinstance(other, type(self)):
            return False
        return self.gte(other)

    def lt(self, other: Measure["Unit"]) -> bool:  # noqa: UP037
        return self.compare_to(other) < 0

    def __lt__(self, other):
        if isinstance(other, (float, int)):
            other = self.unit().of(other)
        if not isinstance(other, type(self)):
            return False
        return self.lt(other)

    def lte(self, other: Measure["Unit"]) -> bool:  # noqa: UP037
        return self.compare_to(other) < 0 or self.is_equivalent(other)

    def __le__(self, other):
        if isinstance(other, (float, int)):
            other = self.unit().of(other)
        if not isinstance(other, type(self)):
            return False
        return self.lte(other)

    @staticmethod
    def max(*measures: "Measure[Unit]"):  # noqa: UP037
        if len(measures) <= 0:
            return None

        max_ = None
        for measure in measures:
            if max_ is None or measure.gt(max_):
                max_ = measure

        return max_

    @staticmethod
    def min(*measures: "Measure[Unit]"):  # noqa: UP037
        if len(measures) == 0:
            return None

        min_ = None
        for measure in measures:
            if min_ is None or measure.lt(min_):
                min_ = measure

        return min_

    def to_short_string(self):
        mag = float(self.magnitude())
        return f"{mag:.3E} {self.unit().symbol()}"

    def to_long_string(self):
        mag = float(self.magnitude())
        if math.isfinite(mag) and mag != 0 and (
            abs(mag) >= 1e7 or abs(mag) < 1e-3
        ):
            significand, exponent = format(
                Decimal(repr(mag)).normalize(), "E"
            ).split("E")
            if "." not in significand:
                significand += ".0"
            return f"{significand}E{int(exponent)} {self.unit().name()}"
        return (
            f"{mag} {self.unit().name()}"
        )

    def __repr__(self):
        return self.to_long_string()


class ImmutableMeasure[U: "Unit"](Measure):
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

    @classmethod
    def of_base_units(
        cls,
        base_unit_magnitude: float,
        unit: U
    ) -> ImmutableMeasure[U]:
        return cls(
            unit.from_base_units(base_unit_magnitude),
            base_unit_magnitude,
            unit,
        )

    @classmethod
    def of_relative_units(
        cls,
        relative_magnitude: float,
        unit: U
    ) -> ImmutableMeasure[U]:
        return cls(
            relative_magnitude, unit.to_base_units(relative_magnitude), unit
        )


class Dimensionless(ImmutableMeasure["Unit"]):
    pass


class Distance(ImmutableMeasure["Unit"]):
    @overload
    def div(self, other: LinearVelocity) -> Time: ...

    @overload
    def div(self, other: Time) -> VelocityUnit[Self]: ...  # type: ignore

    @overload
    def div(self, other: float | Dimensionless) -> Self: ...

    def div(self, other):  # type: ignore
        if isinstance(other, (float, int)):
            return self.base_unit().of(
                self.base_unit_magnitude()
                / other
            )
        res = self.base_unit_magnitude() / other.base_unit_magnitude()
        if isinstance(other, Dimensionless):
            return self.base_unit().of(res)
        elif isinstance(other, LinearVelocity):
            return other.unit().denominator().of(res)
        return super().div(other)


class Time(ImmutableMeasure["Unit"]):
    pass


class Mass(ImmutableMeasure["Unit"]):
    pass


class Angle(ImmutableMeasure["Unit"]):
    @classmethod
    def of_relative_units(
        cls,
        relative_magnitude: float,
        unit: "Unit"  # noqa: UP037
    ) -> Angle:
        return cls(
            relative_magnitude,
            unit.to_base_units(relative_magnitude),
            unit
        )

    @classmethod
    def of_base_units(
        cls,
        base_unit_magnitude: float,
        unit: "Unit"  # noqa: UP037
    ) -> Angle:
        return cls(
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
    def unit(self) -> "PerUnit[DistanceUnit, TimeUnit]":  # noqa: UP037
        return super().unit()  # type: ignore


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


class Acceleration[D: "Unit"](ImmutableMeasure["AccelerationUnit"]):
    pass
