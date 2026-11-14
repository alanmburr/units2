# Copyright (c) FIRST and other WPILib contributors.
# Open Source Software; you can modify and/or share it under the terms of
# the WPILib BSD license file in the root directory of this project.
#
# Contributors:
# - WPILib
# - Alan M. Burr <info@alanburr.us.eu.org>

from __future__ import annotations

import math
import sys
import unittest
from collections.abc import Callable
from typing import (
    Any,
    Final,
    Self,
    cast,
    final,
    overload,
)

from units2 import measure
from units2.collections import LongToObjectHashMap
from units2.measure import Distance, ImmutableMeasure, MeasureOf, Voltage

from .._test import TestCase
from ..units import (
    AngleUnit,
    DistanceUnit,
    LinearVelocityUnit,
    Measure,
    MultUnit,
    PerUnit,
    PowerUnit,
    TimeUnit,
    UnaryFunction,
    Unit,
    UnitBuilder,
    Units,
    VelocityUnit,
    VoltageUnit,
)


class ExampleUnit(Unit):
    def __init__(self, base_unit_equivalent: float):
        super().__init__(
            base_unit=None,
            base_unit_equivalent=base_unit_equivalent,
            unit_name="Example",
            unit_symbol="ex"
        )

    def convert_from(self, magnitude: float, other: Self):
        return self.from_base_units(other.to_base_units(magnitude))

    def of(self, magnitude: float):
        return ImmutableMeasure.of_relative_units(magnitude, self)

    def of_base_units(self, base_unit_magnitude: float):
        return ImmutableMeasure.of_base_units(base_unit_magnitude, self)

    def per(self, other: TimeUnit):  # type: ignore
        return VelocityUnit.combine(self, other)


class ElectricChargeUnit(Unit):
    """ Example new unit from the Readme """

    @overload
    def __init__(self, base_unit_equivalent: float, name: str, symbol: str):
        ...

    @overload
    def __init__(self,
                 to_base_converter: UnaryFunction,
                 from_base_converter: UnaryFunction,
                 name: str, symbol: str): ...

    def __init__(self, bue_tbc, fbc_name, name_symbol, symbol=None):
        if (
            isinstance(bue_tbc, (float, int))
            and isinstance(fbc_name, str)
            and isinstance(name_symbol, str)
        ):
            super().__init__(
                base_unit=None,
                base_unit_equivalent=bue_tbc,
                unit_name=fbc_name,
                unit_symbol=name_symbol
            )
        elif (
            isinstance(bue_tbc, UnaryFunction)
            and isinstance(fbc_name, UnaryFunction)
            and isinstance(name_symbol, str)
            and isinstance(symbol, str)
        ):
            super().__init__(
                base_unit=None,
                to_base_converter=bue_tbc,
                from_base_converter=fbc_name,
                unit_name=name_symbol,
                unit_symbol=symbol
            )
        else:
            raise TypeError(
                "Bad arguments: "
                f"(a) {type(bue_tbc)} should be UnaryFunction|float|int, "
                f"(b) {type(fbc_name)} should be UnaryFunction|str, "
                f"(c) {type(name_symbol)} should be str, "
                f"(d) {type(symbol)} should be str|None"
            )

    def get_base_unit(self) -> Self:
        # The base method must be overridden
        # in order to return the correct type
        return super().get_base_unit()

    def of(self, magnitude: float) -> Measure["ElectricChargeUnit"]:
        return ImmutableMeasure.of_relative_units(magnitude, self)

    def of_base_units(self, magnitude: float) -> Measure["ElectricChargeUnit"]:
        return ImmutableMeasure.of_base_units(magnitude, self)

    def zero(self) -> Measure[Self]:
        return super().zero()

    def one(self) -> Measure[Self]:
        return super().one()

    @overload
    def per(self, other: TimeUnit) -> VelocityUnit[Self]: ...

    @overload
    def per[O: Unit](self, other: O) -> PerUnit[Self, O]: ...

    def per(self, other):
        if isinstance(other, TimeUnit):
            # Note: technically, this would return a CurrentUnit,
            # since electric charge per time is current (measured in Amperes)
            return VelocityUnit.combine(self, other)
        return PerUnit.combine(self, other)

    def convert_from(magnitude: float, other: "ElectricChargeUnit") -> Measure["ElectricChargeUnit"]:
        return self.from_base_units(other.to_base_units(magnitude))


@final
class ElectricChargeUnits:
    Coulomb: Final[ElectricChargeUnit] = ElectricChargeUnit(1.0,
                                                            "Coulomb", "C")
    ElectronCharge: Final[ElectricChargeUnit] = ElectricChargeUnit(1.60217646e-19,
                                                                   "Electron Charge", "e")
    AmpHour: Final[ElectricChargeUnit] = ElectricChargeUnit(3600.0,
                                                            "Amp Hour", "Ah")
    MilliampHour: Final[ElectricChargeUnit] = Units.Milli(AmpHour)


class AnyUnitTest(TestCase):
    def test_unit(self) -> None:
        u = VoltageUnit(base_unit=None, base_unit_equivalent=1.0,
                        unit_name="Volt", unit_symbol="V")
        self.assertEqual("Volt", u.name())
        self.assertEqual("V", u.symbol())
        self.assertEqual(u, Units.Volts)


class CurrentUnitTest(TestCase):
    def test_amps_times_volts(self) -> None:
        combined: PowerUnit = Units.Amps.mult(Units.Volts, "Watt", "W")
        self.assertTrue(combined.equivalent(Units.Watts))

    def test_milliamps_times_millivolts(self) -> None:
        times = Units.Milliamps.mult(Units.Millivolts,
                                     "Microwatt", "uW")
        millimilli = Units.Milli(Units.Milliwatts)

        self.assertEqual(1, times.convert_from(1e-6, Units.Watts))
        self.assertEqual(1, millimilli.convert_from(1e-6, Units.Watts))
        self.assertTrue(times.equivalent(millimilli))


class DistanceUnitTest(TestCase):
    def test_base_unit_distance_per_time(self) -> None:
        anonBaseUnit = (
            DistanceUnit(base_unit=None, base_unit_equivalent=1,
                         unit_name="D", unit_symbol="d")
            .per(TimeUnit(base_unit=None, base_unit_equivalent=1,
                          unit_name="T", unit_symbol="t"))
        )

        self.assertTrue(Units.MetersPerSecond.equivalent(anonBaseUnit))

    def test_feet_per_second(self) -> None:
        ft_per_ms = cast(
            LinearVelocityUnit[DistanceUnit],
            Units.Feet.per(Units.Milliseconds)
        )

        # one foot per millisecond
        # = (1 / 3.28084) meters per (1 / 1000) seconds
        # = (1000 / 3.28084) meters per second
        as_base_measure = ft_per_ms.of(1).to(Units.MetersPerSecond)
        self.assertIsClose(
            1000 / 3.28084, as_base_measure,
            abs_tol=1e-3,
        )

        # one meter per second
        # = 1 mm per millisecond
        # = 0.00328084 feet per millisecond
        as_contrived_measure = (Units.MetersPerSecond.of(1)
                                .to(ft_per_ms))
        self.assertIsClose(
            3.28084 / 1000, as_contrived_measure,
            abs_tol=1e-8
        )


class MeasureTest(TestCase):
    def test_basics(self) -> None:
        unit = Units.Feet
        magnitude = 10
        m = unit.of(magnitude)
        self.assertEqual(unit, m.unit(), "Wrong unit")
        self.assertEqual(magnitude, m.magnitude(), "Wrong magnitude")

    def test_multiply(self) -> None:
        m = Units.Feet.of(1)
        m2 = m.times(10)
        self.assertIsClose(
            10,
            m2.to(Units.Feet),
            abs_tol=1e-12
        )
        self.assertIsNot(m, m2)

    def test_times_conversion_factor(self) -> None:
        m = Units.Feet.of(10)
        f = Units.Feet.of(1)
        d = Units.Degrees.of(10)
        conversion = d.div(f)
        res = m.times_conversion_factor(conversion)
        self.assertEqual(Units.Degrees.of(100), res)

    def test_times_conversion_factor_complex(self) -> None:
        m = Units.Feet.of(1)
        one_foot = Units.Foot.one()
        conversion = (Units.KilogramMetersSquaredPerSecond
                      .of(1)
                      .div(one_foot))
        res = m.times_conversion_factor(conversion)
        self.assertEqual(
            Units.KilogramMetersSquaredPerSecond.of(1),
            res
        )

    def test_times_velocity_conersion_factor(self) -> None:
        m = Units.Seconds.of(10)
        conversion = Units.MetersPerSecond.of(10)
        self.assertIsNotNone(conversion)
        res = m.times_conversion_factor(conversion)
        self.assertEqual(Units.Meters.of(100), res)

    def test_divide(self) -> None:
        m = Units.Meters.of(1)
        m2 = m.div(10)
        self.assertIsClose(
            0.1,
            m2.base_unit_magnitude()
        )
        self.assertIsNot(m2, m)

    def test_add(self) -> None:
        m1 = Units.Feet.of(1)
        m2 = Units.Inches.of(2)
        self.assertTrue(
            m1.plus(m2).is_equivalent(Units.Feet.of(1.0 + 2.0/12.0))
        )
        self.assertTrue(
            m2.plus(m1).is_equivalent(Units.Inches.of(14))
        )

    def test_add_different_dimensions_fails(self) -> None:
        m1 = Units.Inches.one()
        m2 = Units.Radians.one()
        with self.assertRaises(TypeError):
            m1.plus(m2)

    def test_subtract(self) -> None:
        m1 = Units.Feet.of(1)
        m2 = Units.Inches.of(2)
        self.assertTrue(
            m1.minus(m2).is_equivalent(Units.Feet.of(1.0 - 2.0/12.0))
        )
        self.assertTrue(
            m2.minus(m1).is_equivalent(Units.Inches.of(-10))
        )

    def test_subtract_different_dimensions_fails(self) -> None:
        m1 = Units.Inches.one()
        m2 = Units.Radians.one()
        with self.assertRaises(TypeError):
            m1.minus(m2)

    def test_unary_minus(self) -> None:
        m = Units.Feet.of(123)
        negated = m.unary_minus()
        self.assertIsClose(-123, negated.to(Units.Feet), abs_tol=1e-12)
        self.assertEqual(Units.Feet, negated.unit())

    def test_equivalency(self) -> None:
        inches = Units.Inches.of(12)
        feet = Units.Feet.of(1)
        self.assertTrue(inches.is_equivalent(feet))
        self.assertTrue(feet.is_equivalent(inches))

    def test_as(self) -> None:
        m = Units.Inches.of(12)
        self.assertIsClose(
            1,
            m.to(Units.Feet),
            abs_tol=measure.EQUIVALENCE_THRESHOLD
        )

    def test_as_different_dimension_fails(self) -> None:
        m1 = Units.Feet.one()
        with self.assertRaises(TypeError):
            m1.to(Units.Radians)

    def test_per_measure_time(self) -> None:
        m = Units.Kilograms.of(144)
        dt = Units.Milliseconds.of(53)
        res = m.div(dt)
        self.assertIsClose(
            144_000.0 / 53.0,
            res.base_unit_magnitude(),
            abs_tol=1e-5
        )
        self.assertEqual(
            Units.Kilogram.per(Units.Milliseconds),
            res.unit()
        )

    def test_per_unit_time(self) -> None:
        m = Units.Kilograms.of(144)
        res = m.per(Units.Milliseconds)
        self.assertIsInstance(res.unit(), VelocityUnit)
        self.assertIsClose(
            144_000.00,
            res.base_unit_magnitude(),
            abs_tol=1e-5
        )
        self.assertEqual(Units.Kilograms.per(Units.Milliseconds), res.unit())

    def test_divide_measure(self) -> None:
        m1 = Units.Meters.of(6)
        m2 = Units.Value.of(3)
        res = m1.div(m2)
        self.assertEqual(2, res.magnitude())
        self.assertEqual(Units.Meters, res.unit())

        m3 = Units.Meters.of(8)
        m4 = Units.Meters.per(Units.Second).of(4)
        t = m3.div(m4)
        self.assertEqual(2, t.magnitude())
        self.assertEqual(Units.Second, t.unit())

        m5 = Units.Volts.of(6)
        m6 = Units.Volts.per(Units.Meter).of(2)
        d = m5.div(m6)
        self.assertEqual(3, d.magnitude())
        self.assertEqual(Units.Meter, d.unit())

        dist = m5.div(m6)
        self.assertEqual(3, dist.magnitude())
        self.assertEqual(Units.Meter, dist.unit())

    def test_to_short_string(self) -> None:
        m = Units.Volts.of(343)
        self.assertEqual("343.0 Volt", m.to_long_string())
        m2 = Units.Volts.of(343.0001)
        self.assertEqual("343.0001 Volt", m2.to_long_string())
        m3 = Units.Volts.of(123456789.12345678)
        self.assertEqual("1.2345678912345678E8 Volt", m3.to_long_string())

    def test_of_base_units(self) -> None:
        u = ExampleUnit(16)
        m = u.of_base_units(1)
        self.assertEqual(u, m.unit())
        self.assertEqual(1, m.base_unit_magnitude())
        self.assertEqual(1.0/16.0, m.magnitude())

    def test_compare(self) -> None:
        unit = ExampleUnit(7)
        base = unit.of(1)
        less = unit.of(0.5)
        more = unit.of(2)

        self.assertEqual(0, base.compare_to(base))
        self.assertEqual(-1, base.compare_to(more))
        self.assertEqual(1, base.compare_to(less))

        self.assertTrue(base.lt(more))
        self.assertLess(base, more)
        self.assertTrue(base.lte(more))
        self.assertLessEqual(base, more)
        self.assertFalse(base.gt(more))
        self.assertFalse(base > more)
        self.assertFalse(base >= more)

        self.assertTrue(base.gt(less))
        self.assertGreater(base, less)
        self.assertTrue(base.gte(less))
        self.assertGreaterEqual(base, less)
        self.assertFalse(base.lt(less))
        self.assertFalse(base < less)
        self.assertFalse(base.lte(less))
        self.assertFalse(base <= less)

        self.assertLessEqual(base, base)
        self.assertGreaterEqual(base, base)
        self.assertFalse(base < base)  # noqa: PLR0124
        self.assertFalse(base > base)  # noqa: PLR0124

    def test_min_no_args(self) -> None:
        m = Measure.min()
        self.assertIsNone(m)

    def test_min(self) -> None:
        u = ExampleUnit(56.1)
        one = u.of(1)
        two = u.of(2)
        zero = u.of(0)
        very_negative = u.of(-12839712)
        m = Measure.min(one, two, zero, very_negative)
        self.assertIs(very_negative, m)

    def test_min_return_type(self) -> None:
        a = Units.Feet.of(1)
        b = Units.Feet.of(3)
        min_ = Measure.min(a, b)
        self.assertIsInstance(min_, Distance)
        self.assertIs(a, min_)

    def test_max_no_args(self) -> None:
        m = Measure.max()
        self.assertIsNone(m)

    def test_max(self) -> None:
        u = ExampleUnit(6.551)
        one = u.of(1)
        two = u.of(2)
        zero = u.of(0)
        very_large = u.of(8217234)
        m = Measure.max(one, two, zero, very_large)
        self.assertIs(very_large, m)

    def test_max_return_type(self) -> None:
        a = Units.Feet.of(1)
        b = Units.Feet.of(3)
        max_ = Measure.max(a, b)
        self.assertIsInstance(max_, Distance)
        self.assertIs(b, max_)

    def test_is_near_variance_threshold(self) -> None:
        unit = ExampleUnit(92)
        m_a = unit.of(1.21)
        m_b = unit.of_base_units(64)
        # A = 1.21 * 92 base units, or 111.32
        # B = 64 base units
        # ratio = 111.32 / 64 = 1.739375 = 173.9375%

        self.assertTrue(m_a.is_near(m_a, 0))
        self.assertTrue(m_b.is_near(m_b, 0))

        self.assertFalse(m_a.is_near(m_b, 0))
        self.assertFalse(m_a.is_near(m_b, 0.50))
        self.assertFalse(m_a.is_near(m_b, 0.739370))
        self.assertTrue(m_a.is_near(m_b, 0.739375))
        self.assertTrue(m_a.is_near(m_b, 100))

        m_c = unit.of(-1.21)
        m_d = unit.of_base_units(-64)

        self.assertTrue(m_c.is_near(m_c, 0))
        self.assertTrue(m_d.is_near(m_d, 0))

        self.assertFalse(m_c.is_near(m_d, 0))
        self.assertFalse(m_c.is_near(m_d, 0.50))
        self.assertFalse(m_c.is_near(m_d, 0.739370))
        self.assertTrue(m_c.is_near(m_d, 0.739375))
        self.assertTrue(m_c.is_near(m_d, 100))

        m_e = Units.Meters.of(1)
        m_f = Units.Feet.of(-3.28084)

        self.assertTrue(m_e.is_near(m_f, 2.01))
        self.assertFalse(m_e.is_near(m_f, 1.99))

        self.assertTrue(m_f.is_near(m_e, 2.01))
        self.assertFalse(m_f.is_near(m_e, 1.99))

        self.assertTrue(Units.Feet.zero()
                        .is_near(Units.Millimeters.zero(),
                                 0.001))
        self.assertFalse(Units.Feet.of(2)
                         .is_near(Units.Millimeters.of(0),
                                  0.001))

    def test_is_near_measure_tolerance(self) -> None:
        m_compared: Measure
        m_comparing: Measure

        m_compared = Units.Meters.of(1)
        m_comparing = Units.Meters.of(1.2)

        # Positive value with positive tolerance
        self.assertTrue(m_compared.is_near(m_comparing,
                                           Units.Millimeters.of(300)))
        self.assertFalse(m_compared.is_near(m_comparing,
                                            Units.Centimeters.of(10)))

        m_compared = m_compared.unary_minus()
        m_comparing = m_comparing.unary_minus()

        # Negative value with positive tolerance
        self.assertTrue(m_compared.is_near(m_comparing,
                                           Units.Millimeters.of(300)))
        self.assertFalse(m_compared.is_near(m_comparing,
                                            Units.Centimeters.of(10)))

        m_compared = m_compared.unary_minus()
        m_comparing = m_comparing.unary_minus()

        # Positive value with negative tolerance
        self.assertTrue(m_compared.is_near(m_comparing,
                                           Units.Millimeters.of(-300)))
        self.assertFalse(m_compared.is_near(m_comparing,
                                            Units.Centimeters.of(-10)))

        m_compared = m_compared.unary_minus()
        m_comparing = m_comparing.unary_minus()

        # Negative value with negative tolerance.
        self.assertTrue(m_compared.is_near(m_comparing,
                                           Units.Millimeters.of(-300)))
        self.assertFalse(m_compared.is_near(m_comparing,
                                            Units.Centimeters.of(-10)))

        m_compared = m_compared.unary_minus()
        m_comparing = m_comparing.unary_minus()

        # Tolerance exact difference between measures.
        self.assertTrue(m_compared.is_near(m_comparing,
                                           Units.Millimeters.of(200)))
        self.assertTrue(m_compared.is_near(m_comparing,
                                           Units.Centimeters.of(-20)))

    def test_typing(self) -> None:
        arm_mass: MeasureOf.Mass = Units.Kilograms.of(1.423)
        arm_length: MeasureOf.Distance = Units.Inches.of(32.25)
        arm_min_angle: MeasureOf.Angle = Units.Degrees.of(5)
        arm_max_travel: MeasureOf.Angle = Units.Rotations.of(0.45)
        max_speed: MeasureOf.LinearVelocity = Units.MetersPerSecond.of(2.5)

        self.assertIsNotNone(arm_mass)
        self.assertIsNotNone(arm_length)
        self.assertIsNotNone(arm_min_angle)
        self.assertIsNotNone(arm_max_travel)
        self.assertIsNotNone(max_speed)


class MultUnitTest(TestCase):
    def test_automatic_names(self) -> None:
        a = UnitBuilder(ExampleUnit(1)).named("Ay").symbol("a").make()
        b = UnitBuilder(ExampleUnit(1)).named("Bee").symbol("b").make()
        m = MultUnit.combine(a, b)
        self.assertEqual("Ay-Bee", m.name())
        self.assertEqual("a*b", m.symbol())

    def test_combine(self) -> None:
        a = ExampleUnit(100)
        b = ExampleUnit(0.912)
        m = MultUnit.combine(a, b)
        self.assertEqual(91.2, m.to_base_units(1))

    def test_cache(self) -> None:
        a = ExampleUnit(1)
        b = ExampleUnit(2)
        m = MultUnit.combine(a, b)
        self.assertIs(m, MultUnit.combine(a, b))


class PerUnitTest(TestCase):
    def test_of_native(self) -> None:
        part = ExampleUnit(1)
        u = PerUnit.combine(part, part)

        self.assertEqual(10, u.of_native(10).in_unit(u))

        u_of_10 = u.of(10)
        self.assertIsInstance(u_of_10.unit(), PerUnit)
        self.assertEqual("of", cast(PerUnit, u_of_10.unit())._source)
        with self.assertRaises(TypeError):
            u_of_10.in_unit(u)

    def test_mult_same_denom(self) -> None:
        u = Units.Degrees.per(Units.Foot)
        r = u.mult(Units.Foot)
        self.assertIs(Units.Degrees, r)

    def test_mult_other_denom(self) -> None:
        u = Units.Degrees.per(Units.Foot)
        r = u.mult(Units.Inches)
        self.assertIsInstance(r, AngleUnit)
        self.assertIsClose(
            1.0/12.0,
            r.of(1).in_unit(Units.Degrees),
            abs_tol=1e-9
        )
        self.assertEqual("Degree per Foot Inch", r.name())
        self.assertEqual("\xb0/ft-in", r.symbol())


class UnaryFunctionTest(TestCase):
    def test_apply(self) -> None:
        f = UnaryFunction(lambda x: x * x)
        self.assertEqual(1, f.apply(1))
        self.assertEqual(4, f.apply(2))
        self.assertIsClose(
            2,
            f.apply(1.414),  # approx. sqrt(2)
            abs_tol=1e-3
        )
        self.assertEqual(64, f.apply(-8))

    def test_pipe_to(self) -> None:
        f = UnaryFunction(lambda x: x * x)
        g = UnaryFunction(lambda x: x + 1)
        h = f.pipe_to(g)  # h(x) = g(f(x)) = x^2 + 1
        self.assertEqual(2, h.apply(1))
        self.assertEqual(65, h.apply(8))

    def test_mult(self):
        f = UnaryFunction(lambda x: x * x)
        g = UnaryFunction(lambda x: x + 1)
        h = f.mult(g)  # h(x) = f(x) * g(x) = (x^2)(x + 1) = x^3 + x^2
        self.assertEqual(2, h.apply(1))
        self.assertEqual(216 + 36, h.apply(6))
        self.assertEqual(12, h.apply(2))

    def test_mult_scalar(self):
        f = UnaryFunction(lambda x: x * x)
        scalar = 2.5
        h = f.mult(scalar)  # h(x) = 2.5 x^2
        self.assertEqual(250, h.apply(10))
        self.assertEqual(2.5, h.apply(1))
        self.assertEqual(2.5, h.apply(-1))
        self.assertEqual(10, h.apply(2))

    def test_div_zero_denominator(self):
        f = UnaryFunction.IDENTITY
        g = UnaryFunction(lambda _: 0)
        h = f.div(g)
        self.assertEqual(0, h.apply(0))
        self.assertEqual(math.inf, h.apply(1))
        self.assertEqual(-math.inf, h.apply(-1))

    def test_div_nonzero_denominator(self):
        f = UnaryFunction(lambda x: 2 * x)
        g = UnaryFunction(lambda x: x + 1)
        h = f.div(g)
        self.assertEqual(0, h.apply(0))
        self.assertEqual(4, h.apply(-2))
        self.assertEqual(16.0 / 9, h.apply(8))

    def test_div_scalar(self):
        f = UnaryFunction(lambda x: x * x)
        scalar = 2.5
        h = f.div(scalar)  # h(x) = x^2 / 2.5
        self.assertEqual(10, h.apply(5))

    def test_exp(self):
        f = UnaryFunction(lambda x: x * x)
        g = UnaryFunction(lambda x: x + 2)
        h = f.exp(g)  # h(x) = f(x) ^ g(x) = (x^2)^(x + 2)
        self.assertEqual(1, h.apply(1))
        self.assertEqual(0, h.apply(0))
        self.assertEqual(1, h.apply(-2))
        self.assertEqual(256, h.apply(2))  # (2^2)^(2 + 2) = 4^4
        self.assertEqual(59049, h.apply(3))  # (3^2)^(3 + 2) = 9^5

    def test_exp_scalar(self):
        f = UnaryFunction(lambda x: x * x)
        scalar = 4
        h = f.exp(scalar)  # h(x) = f(x)^4 = x^8
        self.assertEqual(0, h.apply(0))
        self.assertEqual(1, h.apply(1))
        self.assertEqual(1, h.apply(-1))
        self.assertEqual(256, h.apply(2))
        self.assertEqual(6561, h.apply(3))


class UnitTest(TestCase):
    def test_of(self) -> None:
        u = ExampleUnit(1)
        five_of_something = u.of(5)
        self.assertIsClose(
            5,
            five_of_something.magnitude(),
            abs_tol=0
        )
        self.assertEqual(u, five_of_something.unit())


class UnitsTest(TestCase):
    ABS_THRESHOLD = 1e-5

    class FloatFunction[C: Any]:
        def __init__(self, f: Callable[[float], C]):
            self._func = f

        def apply(self, x: float):
            return self._func(x)

    def _base_unit_tester(
        self,
        base_unit: Callable[[float], Any] | FloatFunction[Measure]
    ):
        if not isinstance(base_unit, self.FloatFunction):
            base_unit = self.FloatFunction(base_unit)

        self.assertIsClose(
            0,
            base_unit.apply(0).base_unit_magnitude(),
            abs_tol=0
        )
        self.assertIsClose(
            1,
            base_unit.apply(1).base_unit_magnitude(),
            abs_tol=0
        )
        self.assertIsClose(
            -1,
            base_unit.apply(-1).base_unit_magnitude(),
            abs_tol=0
        )
        self.assertIsClose(
            100,
            base_unit.apply(100).base_unit_magnitude(),
            abs_tol=0
        )
        self.assertIsClose(
            8.281723,
            base_unit.apply(8.281723).base_unit_magnitude(),
            abs_tol=0
        )
        self.assertIsClose(
            sys.float_info.max,
            base_unit.apply(sys.float_info.max).base_unit_magnitude(),
            abs_tol=0
        )
        self.assertIsClose(
            sys.float_info.min,
            base_unit.apply(sys.float_info.min).base_unit_magnitude(),
            abs_tol=0
        )

    def test_meters(self) -> None:
        self._base_unit_tester(Units.Meters.of)
        self.assertEqual("Meter", Units.Meters.name())
        self.assertEqual("m", Units.Meters.symbol())

    def test_millimeters(self) -> None:
        self.assertIsClose(1000,
                           Units.Millimeters.convert_from(1, Units.Meters),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertIsClose(1,
                           Units.Meters.convert_from(1000, Units.Millimeters),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Millimeter", Units.Millimeters.name())
        self.assertEqual("mm", Units.Millimeters.symbol())

    def test_centimeters(self) -> None:
        self.assertIsClose(100,
                           Units.Centimeters.convert_from(1, Units.Meters),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertIsClose(1,
                           Units.Meters.convert_from(100, Units.Centimeters),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Centimeter", Units.Centimeters.name())
        self.assertEqual("cm", Units.Centimeters.symbol())

    def test_Inches(self) -> None:
        self.assertIsClose(1, Units.Meters.convert_from(39.3701, Units.Inches),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertIsClose(39.3701,
                           Units.Inches.convert_from(1, Units.Meters),
                           rel_tol=0, abs_tol=1e-4)
        self.assertEqual(1 / 25.4,
                         Units.Inches.convert_from(1, Units.Millimeters))
        self.assertIsClose(12, Units.Inches.convert_from(1, Units.Feet),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Inch", Units.Inches.name())
        self.assertEqual("in", Units.Inches.symbol())

    def test_Feet(self) -> None:
        self.assertIsClose(3.28084, Units.Feet.convert_from(1, Units.Meters),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertIsClose(1 / 12.0, Units.Feet.convert_from(1, Units.Inches),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Foot", Units.Feet.name())
        self.assertEqual("ft", Units.Feet.symbol())

    # Velocities

    def test_meters_per_second(self) -> None:
        self._base_unit_tester(Units.MetersPerSecond.of)
        self.assertEqual("Meter per Second", Units.MetersPerSecond.name())
        self.assertEqual("m/s", Units.MetersPerSecond.symbol())

    def test_FeetPerSecond(self) -> None:
        self.assertIsClose(
            3.28084,
            Units.FeetPerSecond.convert_from(1, Units.MetersPerSecond),
            rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertIsClose(
            1 / 3.28084,
            Units.MetersPerSecond.convert_from(1, Units.FeetPerSecond),
            rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Foot per Second", Units.FeetPerSecond.name())
        self.assertEqual("ft/s", Units.FeetPerSecond.symbol())

    # Accelerations

    def test_MetersPerSecondPerSecond(self) -> None:
        self._base_unit_tester(Units.MetersPerSecondPerSecond.of)
        self.assertEqual("Meter per Second per Second",
                         Units.MetersPerSecondPerSecond.name())
        self.assertEqual("m/s/s", Units.MetersPerSecondPerSecond.symbol())
        self.assertEqual(Units.MetersPerSecond,
                         Units.MetersPerSecondPerSecond.get_unit())
        self.assertEqual(Units.Seconds,
                         Units.MetersPerSecondPerSecond.get_period())

    def test_Gs(self) -> None:
        self.assertIsClose(
            32.17405, Units.Gs.of(1).to(Units.FeetPerSecond.per(Units.Second)),
            rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertIsClose(
            9.80665, Units.Gs.of(1).to(Units.MetersPerSecondPerSecond),
            rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("G", Units.Gs.name())
        self.assertEqual("G", Units.Gs.symbol())
        self.assertEqual(Units.MetersPerSecond, Units.Gs.get_unit())
        self.assertEqual(Units.Seconds, Units.Gs.get_period())

    # Time

    def test_Seconds(self) -> None:
        self._base_unit_tester(Units.Seconds.of)
        self.assertEqual("Second", Units.Seconds.name())
        self.assertEqual("s", Units.Seconds.symbol())

    def test_Millisecond(self) -> None:
        self.assertIsClose(1000,
                           Units.Milliseconds.convert_from(1, Units.Seconds),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Millisecond", Units.Milliseconds.name())
        self.assertEqual("ms", Units.Milliseconds.symbol())

    def test_Microsecond(self) -> None:
        self.assertEqual(1e6,
                         Units.Microseconds.convert_from(1, Units.Second))
        self.assertEqual("Microsecond", Units.Microseconds.name())
        self.assertEqual("us", Units.Microseconds.symbol())

    def test_Minutes(self) -> None:
        self.assertIsClose(60, Units.Seconds.convert_from(1, Units.Minutes),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Minute", Units.Minutes.name())
        self.assertEqual("min", Units.Minutes.symbol())

    # Mass

    def test_Kilograms(self) -> None:
        self._base_unit_tester(Units.Kilograms.of)
        self.assertEqual("Kilogram", Units.Kilograms.name())
        self.assertEqual("Kg", Units.Kilograms.symbol())

    def test_Grams(self) -> None:
        self.assertIsClose(1000, Units.Grams.convert_from(1, Units.Kilograms),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Gram", Units.Grams.name())
        self.assertEqual("g", Units.Grams.symbol())

    def test_Pounds(self) -> None:
        self.assertIsClose(453.592, Units.Grams.convert_from(1, Units.Pounds),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Pound", Units.Pounds.name())
        self.assertEqual("lb.", Units.Pounds.symbol())

    def test_Ounces(self) -> None:
        self.assertIsClose(16, Units.Ounces.convert_from(1, Units.Pounds),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Ounce", Units.Ounces.name())
        self.assertEqual("oz.", Units.Ounces.symbol())

    # Angle

    def test_Revolutions(self) -> None:
        self.assertIsClose(
            1, Units.Revolutions.convert_from(2 * math.pi, Units.Radians),
            rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Revolution", Units.Revolutions.name())
        self.assertEqual("R", Units.Revolutions.symbol())

    def test_Radians(self) -> None:
        self._base_unit_tester(Units.Radians.of)
        self.assertIsClose(
            2 * math.pi, Units.Radians.convert_from(1, Units.Revolutions),
            rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertIsClose(
            2 * math.pi, Units.Radians.convert_from(360, Units.Degrees),
            rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Radian", Units.Radians.name())
        self.assertEqual("rad", Units.Radians.symbol())

    def test_Degrees(self) -> None:
        self.assertIsClose(360,
                           Units.Degrees.convert_from(1, Units.Revolutions),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertIsClose(
            360, Units.Degrees.convert_from(2 * math.pi, Units.Radians),
            rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Degree", Units.Degrees.name())
        self.assertEqual("°", Units.Degrees.symbol())

    # Unitless

    def test_Value(self) -> None:
        self._base_unit_tester(Units.Value.of)
        self.assertEqual("<?>", Units.Value.name())
        self.assertEqual("<?>", Units.Value.symbol())

    def test_Percent(self) -> None:
        self.assertIsClose(100, Units.Percent.convert_from(1, Units.Value),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Percent", Units.Percent.name())
        self.assertEqual("%", Units.Percent.symbol())

    # Electric potential

    def test_Volts(self) -> None:
        self._base_unit_tester(Units.Volts.of)
        self.assertEqual("Volt", Units.Volts.name())
        self.assertEqual("V", Units.Volts.symbol())

    def test_Millivolts(self) -> None:
        self.assertIsClose(1000, Units.Millivolts.convert_from(1, Units.Volts),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Millivolt", Units.Millivolts.name())
        self.assertEqual("mV", Units.Millivolts.symbol())

    # Electric current

    def test_Amps(self) -> None:
        self._base_unit_tester(Units.Amps.of)
        self.assertEqual("Amp", Units.Amps.name())
        self.assertEqual("A", Units.Amps.symbol())

    def test_Milliamps(self) -> None:
        self.assertIsClose(1000, Units.Milliamps.convert_from(1, Units.Amps),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Milliamp", Units.Milliamps.name())
        self.assertEqual("mA", Units.Milliamps.symbol())

    # Power

    def test_Watts(self) -> None:
        self._base_unit_tester(Units.Watts.of)
        self.assertEqual("Watt", Units.Watts.name())
        self.assertEqual("W", Units.Watts.symbol())

    def test_Milliwatts(self) -> None:
        self.assertIsClose(1000, Units.Milliwatts.convert_from(1, Units.Watts),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Milliwatt", Units.Milliwatts.name())
        self.assertEqual("mW", Units.Milliwatts.symbol())

    def test_Horsepower(self) -> None:
        self.assertIsClose(745.7,
                           Units.Watts.convert_from(1, Units.Horsepower),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertEqual("Horsepower", Units.Horsepower.name())
        self.assertEqual("HP", Units.Horsepower.symbol())

    # Temperature

    def test_Kelvin(self) -> None:
        self._base_unit_tester(Units.Kelvin.of)
        self.assertEqual("Kelvin", Units.Kelvin.name())
        self.assertEqual("K", Units.Kelvin.symbol())

    def test_Celsius(self) -> None:
        self.assertIsClose(0, Units.Celsius.of(-273.15).to(Units.Kelvin),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertIsClose(273.15, Units.Celsius.of(0).to(Units.Kelvin),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertIsClose(0, Units.Kelvin.of(273.15).to(Units.Celsius),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertTrue(Units.Celsius.of(0).is_equivalent(
            Units.Kelvin.of(273.15)))
        self.assertTrue(Units.Celsius.of(-273.15).is_equivalent(
            Units.Kelvin.of(0)))
        self.assertEqual("Celsius", Units.Celsius.name())
        self.assertEqual("\xb0C", Units.Celsius.symbol())

    def test_Fahrenheit(self) -> None:
        self.assertIsClose(0, Units.Fahrenheit.of(32).to(Units.Celsius),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertIsClose(100, Units.Fahrenheit.of(212).to(Units.Celsius),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertIsClose(-459.67, Units.Kelvin.of(0).to(Units.Fahrenheit),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertIsClose(273.15, Units.Fahrenheit.of(32).to(Units.Kelvin),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertIsClose(32, Units.Kelvin.of(273.15).to(Units.Fahrenheit),
                           rel_tol=0, abs_tol=self.ABS_THRESHOLD)
        self.assertTrue(Units.Fahrenheit.of(32).is_equivalent(
            Units.Celsius.of(0)))
        self.assertTrue(Units.Fahrenheit.of(212).is_equivalent(
            Units.Celsius.of(100)))
        self.assertEqual("Fahrenheit", Units.Fahrenheit.name())
        self.assertEqual("\xb0F", Units.Fahrenheit.symbol())

    # Helpers

    def test_Kilo(self) -> None:
        u = ExampleUnit(1)
        kiloUnit = Units.Kilo(u)
        self.assertEqual(1e3, u.convert_from(1, kiloUnit), 0)
        self.assertEqual(1e-3, kiloUnit.convert_from(1, u), 0)

    def test_Milli(self) -> None:
        u = ExampleUnit(1)
        milli_u = Units.Milli(u)
        self.assertEqual(1e-3, u.convert_from(1, milli_u), 0)
        self.assertEqual(1e3, milli_u.convert_from(1, u), 0)


class VelocityUnitTest(TestCase):
    def test_base_unit(self) -> None:
        self.assertTrue(Units.MetersPerSecond.equivalent(
            Units.MetersPerSecond))
        self.assertTrue(Units.Meters.per(Units.Second).equivalent(
            Units.MetersPerSecond))

    def test_to_acceleration(self) -> None:
        u = Units.MetersPerSecond.per(Units.Millisecond)
        self.assertEqual(1000, u.of(1).to(Units.MetersPerSecondPerSecond))
        self.assertEqual(0, u.of(0).to(Units.MetersPerSecondPerSecond))

    def test_cache(self) -> None:
        self.assertIs(
            Units.FeetPerSecond, Units.Feet.per(Units.Second),
            "Feet.per(Second) should return a cached object instance")
        dist = AngleUnit(
            base_unit=Units.Radians, base_unit_equivalent=123,
            unit_name="a", unit_symbol="a")
        time = TimeUnit(
            base_unit=Units.Seconds, base_unit_equivalent=123,
            unit_name="t", unit_symbol="t")
        first = dist.per(time)
        second = dist.per(time)
        self.assertIs(
            first,
            second,
            f"{first} was not the same object as {second}")

    def test_mult(self) -> None:
        baseUnit = ExampleUnit(92)
        vel = VelocityUnit.combine(baseUnit, Units.Millisecond)
        mult = vel.mult(Units.Minute)
        self.assertIsClose(1 / 92000.0, mult.from_base_units(1),
                           abs_tol=1e-5)
        self.assertIsClose(5_520_000, mult.to_base_units(1),
                           abs_tol=1e-5)


class VoltageUnitTest(TestCase):
    def test_volts_times_amps(self) -> None:
        self.assertTrue(Units.Volts.mult(Units.Amps, "", "")
                        .equivalent(Units.Watts))

    def test_millivolts_times_milliamps(self) -> None:
        self.assertTrue(Units.Millivolts.mult(Units.Milliamps, "", "")
                        .equivalent(Units.Micro(Units.Watts)))


class Measure_VelocityTest(TestCase):
    def test_velocity_times_time_returns_divisor(self) -> None:
        v = VelocityUnit.combine(Units.Volts, Units.Seconds).of(123)
        t = Units.Milliseconds.of(100)
        r = v.times(t)
        self.assertIsInstance(r, Voltage)


class Collections_LongToObjectHashMapTest(TestCase):
    def test_put(self) -> None:
        map = LongToObjectHashMap[str]()
        self.assertEqual(0, map.size())

        key = 42
        prev_val = map.put(key, "The Meaning of Life")
        self.assertIsNone(prev_val)
        self.assertEqual(1, map.size())
        self.assertEqual("The Meaning of Life", map.get(key))
        self.assertTrue(map.contains_key(key))

        prev_val = map.put(key, "Addendum: Live Long and Prosper")
        self.assertEqual("The Meaning of Life", prev_val)
        self.assertEqual(1, map.size(),
                         "The map should not have grown when changing the "
                         "value for a key")
        self.assertTrue(map.contains_key(key))

        removed = map.remove(key)
        self.assertEqual("Addendum: Live Long and Prosper", removed)
        self.assertEqual(0, map.size())
        self.assertFalse(map.contains_key(key))

    def test_growth(self) -> None:
        map = LongToObjectHashMap[str]()
        n_inserts = 8000
        for i in range(n_inserts):
            map.put(i, f"value-{i}")
        self.assertEqual(n_inserts, map.size())
        self.assertEqual(16384, map.capacity())

        for i in range(n_inserts):
            self.assertEqual(f"value-{i}", map.get(i),
                             "key {i} had an unexpected value")
            self.assertTrue(map.contains_key(i))

        for i in range(n_inserts):
            map.remove(i)

        # all items should have been removed
        self.assertEqual(0, map.size())

        # ... but the map shouldn't resize
        self.assertEqual(16384, map.capacity())

    def test_clear(self) -> None:
        map = LongToObjectHashMap[int]()
        for i in range(10_000):
            map.put(i, i * i)
        self.assertEqual(10_000, map.size())
        self.assertEqual(16384, map.capacity())

        map.clear()
        self.assertEqual(0, map.size())
        self.assertEqual(16384, map.capacity())

        map.clear()  # should be a NOP
        self.assertEqual(0, map.size())
        self.assertEqual(16384, map.capacity())

    def test_key_set(self) -> None:
        map = LongToObjectHashMap[str]()
        n_inserts = 8000
        for i in range(n_inserts):
            map.put(i, f"value-{i}")

        key_set = map.key_set()
        self.assertEqual(map.size(), key_set.size())
        for i in range(n_inserts):
            self.assertTrue(key_set.contains(i),
                            f"keySet does not contain key {i}")


class AngleTest(TestCase):
    def test_cos(self) -> None:
        m1 = Units.Radians.of(math.pi)
        m2 = Units.Degrees.of(180)
        self.assertEqual(m1, m2)
        self.assertIsClose(m1.cos(), -1,
                           abs_tol=1e-5)
        self.assertIsClose(m2.cos(), -1,
                           abs_tol=1e-5)

    def test_sec(self) -> None:
        m = Units.Degrees.of(30)
        two_thirds_sqrt3 = (2.0/3.0) * math.sqrt(3.0)
        self.assertEqual(m.sec(), two_thirds_sqrt3)

    def test_sin(self) -> None:
        m1 = Units.Radians.of(math.pi/2.0)
        m2 = Units.Degrees.of(90)
        self.assertEqual(m1, m2)
        self.assertEqual(m1.sin(), 1)
        self.assertEqual(m2.sin(), 1)

    def test_csc(self) -> None:
        m = Units.Degrees.of(30)
        self.assertIsClose(m.csc(), 2,
                           abs_tol=1e-5)

    def test_tan(self) -> None:
        m1 = Units.Radians.of(math.pi/4.0)
        m2 = Units.Degrees.of(45)
        self.assertEqual(m1, m2)
        self.assertEqual(m1, m2)
        self.assertIsClose(m1.tan(), 1,
                           abs_tol=1e-5)
        self.assertIsClose(m2.tan(), 1,
                           abs_tol=1e-5)

    def test_cot(self) -> None:
        m = Units.Degrees.of(30)
        sqrt3 = math.sqrt(3.0)
        self.assertIsClose(m.cot(), sqrt3,
                           abs_tol=1e-5)


class AccelerationUnitTest(TestCase):
    def test_amps_per_second_per_second(self) -> None:
        u = Units.Amps.per(Units.Second).per(Units.Second)
        m = u.of(0.25)
        self.assertEqual("Amp per Second per Second", u.name())
        self.assertEqual("A/s/s", u.symbol())
        self.assertEqual(0.25, m.magnitude())


class ElectricChargeTest(TestCase):
    def test_coulomb_per_second_equals_amps(self) -> None:
        c = Units.Amps.of(30)
        u = ElectricChargeUnits.Coulomb.per(Units.Seconds)
        q_by_t = u.of(30)
        self.assertEqual("Coulomb per Second", u.name())
        self.assertEqual("C/s", u.symbol())
        self.assertEqual(c.magnitude(), q_by_t.magnitude())

    def test_elementary_charge_times_volts_equals_joules(self) -> None:
        m = ElectricChargeUnits.ElectronCharge.one().times(Units.Volts.of(100))
        self.assertEqual("Electron Charge-Volt", m.unit().name())
        self.assertEqual("e*V", m.unit().symbol())
        self.assertIsClose(1.60218e-17, m.base_unit_magnitude(), abs_tol=1e-5)

    def test_current_times_time_equals_amp_hour(self) -> None:
        m = ElectricChargeUnits.AmpHour.of(6)
        m2 = Units.Amps.of(3).times(Units.Hours.of(2))
        self.assertEqual("Amp-Hour", m2.unit().name())
        self.assertEqual("A*hr", m2.unit().symbol())
        self.assertEqual(m2.magnitude(), m.magnitude())


if __name__ == "__main__":
    unittest.main()
