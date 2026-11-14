# Copyright (c) FIRST and other WPILib contributors.
# Open Source Software; you can modify and/or share it under the terms of
# the WPILib BSD license file in the root directory of this project.
#
# Contributors:
# - WPILib
# - Alan M. Burr <info@alanburr.us.eu.org>

import sys

if sys.version_info < (3, 12):  # noqa: UP036
    raise RuntimeError(
        "units2 requires python>=3.12"
        ", you have {}".format(sys.version)
    )

from .measure import MeasureOf
from .units import Units

__all__ = [
    "MeasureOf",
    "Units"
]
