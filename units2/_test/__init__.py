import unittest

from ..units import PowerUnit, Units


class CurrentUnitTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
