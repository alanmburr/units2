# units2: WPILib Units Library - Python port

This readme has been adapted from the ["Java Units Library" page published by WPILib](https://docs.wpilib.org/en/stable/docs/software/basic-programming/java-units.html).

This library was ported from Java to Python and implements the same API to the user, except in cases where methods (such as `in`) could not work becuase their names are reserved keywords in Python. To preserve the Python spirit, all method names were converted from camelCase to snake_case, but are invoked in the same way (examples below).

## Version Information & Installation
Requires ``python>=3.12``, recommends ``python>=3.14``.

To install,
```shell
pip install git+https://github.com/alanmburr/units2.git
```

---

The units library is a tool that helps programmers avoid mistakes related to units of measurement. It does this by keeping track of the units of measurement, and by ensuring that all operations are performed with the correct units. This can help to prevent errors that can lead to incorrect results, such as adding a distance in inches to a distance in meters.

An added benefit is readability and maintainability, which also reduces bugs. By making the units of measurement explicit in your code, it becomes easier to read and understand what your code is doing. This can also help to make your code more maintainable, as it is easier to identify and fix errors related to units of measurement.

The units library has a number of features:

- A set of predefined units, such as meters, degrees, and seconds.
- The ability to convert between different units.
- Support for performing arithmetic and comparisons on quantities with units.
- Support for displaying quantities with units in a human-readable format.

## Terminology

**Dimension** - Dimensions represent the nature of a physical quantity, such as length, time, or mass. They are independent of any specific unit system. For example, the dimension of meters is length, regardless of whether the length is expressed in meters, millimeters, or inches.

**Unit** - Units are specific realizations of dimensions. They are the way of expressing physical quantities. Each dimension has a base unit, such as the meter for length, the second for time, the kilogram for mass. Derived units are formed by combining base units, such as meters per second for velocity.

**Measure** - Measures are the specific magnitude of physical quantities, expressed in a particular unit. For example, 5 meters is a measure of distance.

These concepts are used within the Units Library. For example, the **measure** *10 seconds* has a magnitude of 10, the **dimension** is time, and the **unit** is seconds.

## Using the Units Library

The Python units library is available in the ``units2`` package. The most relevant classes are:

- The various classes for predefined dimensions, such as DistanceUnit and TimeUnit
- Units, which contains a set of predefined units. Take a look a the [Units javadoc](https://github.wpilib.org/allwpilib/docs/beta/java/wpilib.units/org/wpilib/units/Units.html) to browse the available units and their types. *All units available in Java are available in Python as well.*
- [Measure](https://github.wpilib.org/allwpilib/docs/beta/java/wpilib.units/org/wpilib/units/Measure.html), which is used to tag a value with a unit, and the dimension-specific implementations like Distance and Time

> [!IMPORTANT]
> Unlike in the Java API, you cannot static import all units, you'll have to `from units2 import Units` and then refer to units as `Units.<unit>`, for example ``Units.Amps``. Future releases may expose a similar API without the need for importing all units. Similarly *to Java*, ``Measure``s can be accessed by ``from units2.measure import <measure>``.

### Creating Measures

Every dimension has a measurement class with the corresponding name - for example, a ``Distance`` measures distance, ``Time`` measures time, and ``LinearVelocity`` measures linear velocity. To instantiate one of these measurements, call the ``Unit.of`` method on the appropriate unit object. For example, to create a ``Distance`` object representing a distance of 6 inches, you would write:

```python
wheel_diameter: Distance = Units.Inches.of(6)
```

Other measures can also be created using their ``Unit.of`` method:

```python
# from units2 import MeasureOf
ARM_MASS: MeasureOf.Mass = Units.Kilograms.of(1.423)
ARM_LENGTH: MeasureOf.Distance = Units.Inches.of(32.25)
ARM_MIN_ANGLE: MeasureOf.Angle = Units.Degrees.of(5)
ARM_MAX_TRAVEL: MeasureOf.Angle = Units.Rotations.of(0.45)
MAX_SPEED: MeasureOf.LinearVelocity = Units.MetersPerSecond.of(2.5)
# Your static type checker may not like these annotations, but they are valid;
# just be sure to `from __future__ import annotations`.

# Which is the same as
# from units2.measure import Mass, Distance, Angle, LinearVelocity
ARM_MASS: Mass = Units.Kilograms.of(1.423)
ARM_LENGTH: Distance = Units.Inches.of(32.25)
ARM_MIN_ANGLE: Angle = Units.Degrees.of(5)
ARM_MAX_TRAVEL: Angle = Units.Rotations.of(0.45)
MAX_SPEED: LinearVelocity = Units.MetersPerSecond.of(2.5)
# Your static type checker may not like these annotations, but they are valid;
# just be sure to `from __future__ import annotations`.
```

> [!WARNING]
> Composite units with ``PerUnit`` and ``MultUnit`` have special requirements, and the ``of`` method is not recommended to be used with them. Inspect the PerUnit and MultUnit classes before inheriting from them, as they have
non-bodied methods that will `raise` when called. 

#### Using Composite Unit Types

Due to requirements of inheritance in Python's type system, ``PerUnit`` and ``MultUnit`` cannot return a normal ``Per`` or ``Mult`` object from their ``of`` factory methods. Instead, they need to return a bounded wildcard ``Measure[PerUnit]`` or ``Measure[MultUnit]``  (notice the lack of greater type specificity on ``PerUnit`` and ``MultUnit``) to allow subclasses like ``LinearVelocity`` to return a compatible type. New ``of_native`` methods are provided to be able to work with known ``Per`` and ``Mult`` objects.

```python
# Assuming a well-typed VoltsPerMeter implementation that doesn't exist on Units

# Using of_native:
kP: Per[VoltageUnit, DistanceUnit] = VoltsPerMeter.of_native(1)
kP.in_unit(VoltsPerMeter); # 1.0

output: Measure[VoltageUnit] = kP.times_divisor(Units.Meters.of(1))
output.in_unit(Volts);  # 1.0

# Without of_native
kP: Measure[PerUnit[VoltageUnit, DistanceUnit]] = VoltsPerMeter.of(1)
kP.in_unit(VoltsPerMeter);  # Runtime TypeError!

# Below is not valid in Java, but it is in Python; be careful!
# Static type checker doesn't know what unit this is, but it will exist.
# Depending on what you pass to times, you may get a useful value or 
# maybe your program will raise (throw).
output: Measure = kP.times(Meters.of(1))
output.in_unit(Units.Volts)
```

### Performing Calculations

The ``Measure`` class also supports arithmetic operations, such as addition, subtraction, multiplication, and division. These are done by calling methods on the objects. These operations always ensure that the units are compatible before performing the calculation, and they return a new ``Measure`` object. For example, you can add two ``Distance`` objects together, even if they have different units:

```python
distance1: MeasureOf.Distance = Units.Inches.of(10)
distance2: MeasureOf.Distance = Units.Meters.of(0.254)
total_distance: MeasureOf.Distance = distance1.plus(distance2)
```

In this code, the units library will automatically convert the measures to the same unit before adding the two distances. The resulting ``total_distance`` object will be a new ``Distance`` object that has a value of 0.508 meters, or 20 inches.

> [!NOTE]
> Mathematical operations are type safe (to an extent; your program will `raise` if you try to add or subtract dimensionally incomparable units). It is impossible to add a distance to a time, or subtract an angle from a voltage. However, multiplication and division operations make a best-effort attempt to return results in the most appropriate unit type; dividing a distance by time results in a ``LinearVelocity`` measurement, and multiplying it by time returns a ``Distance``.

This example combines the wheel diameter and gear ratio to calculate the distance per rotation of the wheel:

```python
wheel_diameter: MeasureOf.Distance = Units.Inches.of(3)
double gear_ratio = 10.48
distancePerRotation: MeasureOf.Distance = wheel_diameter.times(math.pi).div(gear_ratio)
```

> [!IMPORTANT]
> By default, arithmetic operations create new ``Measure`` instances for their results. See Read more online about Python's garbabe collection to learn about the impact of creating a large number of short-lived objects.

### Converting Units

Unit conversions can be done by calling ``Measure.in_unit(Unit)`` (note that this is different from Java because the `in` keyword is reserved in Python). The library should prevent units from being converted between incompatible types, such as distances to angles. The returned values will be bare ``float`` values without unit information - it is up to you, the programmer, to interpret them correctly! It is strongly recommended to only use unit conversions when interacting with APIs that do not support the units library.

```python
MAX_VELOCITY: LinearVelocity = Units.FeetPerSecond.of(12.5)
MAX_ACCELERATION: LinearAcceleration = Units.FeetPerSecond.per(Second).of(22.9)
MAX_VELOCITY.in_unit(Units.MetersPerSecond);  # => OK! Returns 3.81
MAX_VELOCITY.in_unit(Units.RadiansPerSecond);  # => Runtime TypeError! LinearVelocity cannot be converted to AngularVelocity

# The WPILib math libraries use SI metric units, so we have to convert to meters:
DRIVE_CONSTRAINTS = TrapezoidProfile.Constraints(
  MAX_VELOCITY.in_unit(Units.MetersPerSecond),
  MAX_ACCELERATION.in_unit(Units.MetersPerSecondPerSecond)
)
```

### Usage Example

Pulling all of the concepts together, we can create an example that calculates the end effector position of an arm mechanism:

```python
# def get_arm_angle(...) -> Angle: ...

arm_length: MeasureOf.Distance = Units.Feet.of(3).plus(Units.Inches.of(4.25))
end_effector_x: MeasureOf.Distance = arm_length.times(get_arm_angle().cos()))
end_effector_y: MeasureOf.Distance = arm_length.times(get_arm_angle().sin()))
```

### Human-readable Formatting

The ``Measure`` class has methods that can be used to get a human-readable representation of the measure. This feature is useful to display a measure on a dashboard or in logs.

- ``str(<measure>)`` and ``.to_short_string()`` return a string representation of the measure in a shorthand form. The symbol of the backing unit is used, rather than the full name, and the magnitude is represented in scientific notation. For example, "1.234e+04 V/m"
- ``.to_long_string()`` returns a string representation of the measure in a longhand form. The name of the backing unit is used, rather than its symbol, and the magnitude is represented in a full string, not scientific notation. For example, "1234 Volt per Meter"

## Defining New Units

There are four ways to define a new unit that isn't already present in the library:

- Using the ``Unit.per`` or ``Unit.mult`` methods to create a composite of two other units;
- Using the ``Milli``, ``Micro``, and ``Kilo`` Units helper methods;
- Using the ``Unit.derive`` method and customizing how the new unit relates to the base unit; and
- Subclassing ``Unit`` to define a new dimension.

New units can be defined as combinations of existing units using the ``Unit.mult`` and ``Unit.per`` methods.

```python
VoltsPerInch: PerUnit[VoltageUnit, DistanceUnit] = Units.Volts.per(Units.Inch)
KgPerSecond: VelocityUnit[MassUnit] = Kilograms.per(Units.Second)  # Could also be declared as PerUnit[MassUnit, TimeUnit]
FootMinutesPerSecond: DistanceUnit = FeetPerSecond.mult(Units.Minutes)
```

Using ``mult`` and ``per`` will store the resulting unit. Every call will return the same object to avoid unnecessary allocations and garbage collector pressure.

```python
def robotPeriodic():
  # Units.Feet.per(Units.Millisecond) creates a new unit on the first loop,
  # which will be reused on every successive loop
  SmartDashboard.putNumber("Speed", _drivebase.get_speed().in_unit(Units.Feet.per(Units.Millisecond)))
```

> [!NOTE]
> Calling ``Unit.per(Time)`` will return a ``Velocity`` unit, which is different from and incompatible with a ``Per`` unit!

New dimensions can also be created by subclassing ``Unit`` and implementing the two constructors. Dimension-specific measurement types are recommended, but take considerable effort to implement all the unit-specific math operations.

```python
# from units2.units import UnaryFunction, TimeUnit, VelocityUnit, PerUnit
# from units2.measure import Measure, ImmutableMeasure
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


# from units2._test import TestCase
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
```

> [!NOTE]
> Your static type checker might not like the way methods are overridden (e.g., ``.of()``, ``.one()``) but as long as you write tests to verify that your unit works, you can safely ignore those (and silence them by using ``# type: ignore`` for Mypy and Pylance and ``# noqa`` for Ruff and Sonarqube).
