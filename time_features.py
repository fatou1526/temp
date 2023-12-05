#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 00:36:57 2023

@author: test
"""

# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Directory to extract time covariates.

Extract time covariates from datetime.
"""

import numpy as np
import pandas as pd
from pandas.tseries.holiday import EasterMonday
from pandas.tseries.holiday import GoodFriday
from pandas.tseries.holiday import Holiday
from pandas.tseries.holiday import SU
from pandas.tseries.holiday import TH
from pandas.tseries.holiday import USColumbusDay
from pandas.tseries.holiday import USLaborDay
from pandas.tseries.holiday import USMartinLutherKingJr
from pandas.tseries.holiday import USMemorialDay
from pandas.tseries.holiday import USPresidentsDay
from pandas.tseries.holiday import USThanksgivingDay
from pandas.tseries.offsets import DateOffset
from pandas.tseries.offsets import Day
from pandas.tseries.offsets import Easter
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# This is 183 to cover half a year (in both directions), also for leap years
# + 17 as Eastern can be between March, 22 - April, 25
MAX_WINDOW = 183 + 17


def _distance_to_holiday(holiday):
  """Return distance to given holiday."""

  def _distance_to_day(index):
    holiday_date = holiday.dates(
        index - pd.Timedelta(days=MAX_WINDOW),
        index + pd.Timedelta(days=MAX_WINDOW),
    )
    assert (
        len(holiday_date) != 0  # pylint: disable=g-explicit-length-test
    ), f"No closest holiday for the date index {index} found."
    # It sometimes returns two dates if it is exactly half a year after the
    # holiday. In this case, the smaller distance (182 days) is returned.
    return (index - holiday_date[0]).days

  return _distance_to_day


from pandas.tseries.holiday import Holiday, DateOffset, Easter, MO, SU

# Vacances pour Sierra Leone
EasterSunday = Holiday("Easter Sunday", month=1, day=1, offset=[Easter(), DateOffset(weekday=SU(0))])
NewYearsDay = Holiday("New Years Day", month=1, day=1)
IndependenceDay = Holiday("Independence Day", month=4, day=27)
ChristmasEve = Holiday("Christmas Eve", month=12, day=24)
ChristmasDay = Holiday("Christmas Day", month=12, day=25)
BoxingDay = Holiday("Boxing Day", month=12, day=26)
NewYearsEve = Holiday("New Years Eve", month=12, day=31)

# Ajout d'autres vacances
ArmedForcesDay = Holiday("Armed Forces Day", month=2, day=18)
LabourDay = Holiday("Labour Day", month=5, day=1)
#MuslimFestival = Holiday("Muslim Festival", month=varies, day=varies)  Remplacez les valeurs par les dates réelles
#IslamicNewYear = Holiday("Islamic New Year", month=varies, day=varies)  Remplacez les valeurs par les dates réelles

HOLIDAYS = [
    EasterSunday,
    NewYearsDay,
    IndependenceDay,
    ChristmasEve,
    ChristmasDay,
    NewYearsEve,
    BoxingDay,
    ArmedForcesDay,
    LabourDay
]


class TimeCovariates(object):
  """Extract all time covariates except for holidays."""

  def __init__(
      self,
      datetimes,
      normalized = False,
      holiday = False,
  ):
    """Init function.

    Args:
      datetimes: pandas DatetimeIndex (lowest granularity supported is min)
      normalized: whether to normalize features or not
      holiday: fetch holiday features or not

    Returns:
      None
    """
    self.normalized = normalized
    self.dti = datetimes
    self.holiday = holiday

  """
  def _minute_of_hour(self):
    minutes = np.array(self.dti.minute, dtype=np.float32)
    if self.normalized:
      minutes = minutes / 59.0 - 0.5
    return minutes

  def _hour_of_day(self):
    hours = np.array(self.dti.hour, dtype=np.float32)
    if self.normalized:
      hours = hours / 23.0 - 0.5
    return hours
  """
  def _day_of_week(self):
    day_week = np.array(self.dti.dayofweek, dtype=np.float32)
    if self.normalized:
      day_week = day_week / 6.0 - 0.5
    return day_week

  def _day_of_month(self):
    day_month = np.array(self.dti.day, dtype=np.float32)
    if self.normalized:
      day_month = day_month / 30.0 - 0.5
    return day_month

  def _day_of_year(self):
    day_year = np.array(self.dti.dayofyear, dtype=np.float32)
    if self.normalized:
      day_year = day_year / 364.0 - 0.5
    return day_year

  def _month_of_year(self):
    month_year = np.array(self.dti.month, dtype=np.float32)
    if self.normalized:
      month_year = month_year / 11.0 - 0.5
    return month_year

  def _week_of_year(self):
    week_year = np.array(self.dti.strftime("%U").astype(int), dtype=np.float32)
    if self.normalized:
      week_year = week_year / 51.0 - 0.5
    return week_year

  def _get_holidays(self):
    dti_series = self.dti.to_series()
    hol_variates = np.vstack(
        [
            dti_series.apply(_distance_to_holiday(h)).values
            for h in tqdm(HOLIDAYS)
        ]
    )
    return StandardScaler().fit_transform(hol_variates)

  def get_covariates(self):
    """Get all time covariates."""
    #moh = self._minute_of_hour().reshape(1, -1)
    #hod = self._hour_of_day().reshape(1, -1)
    dom = self._day_of_month().reshape(1, -1)
    dow = self._day_of_week().reshape(1, -1)
    doy = self._day_of_year().reshape(1, -1)
    moy = self._month_of_year().reshape(1, -1)
    woy = self._week_of_year().reshape(1, -1)

    all_covs = [
        dom,
        dow,
        doy,
        moy,
        woy,
    ]
    columns = ["dom", "dow", "doy", "moy", "woy"]
    if self.holiday:
      hol_covs = self._get_holidays()
      all_covs.append(hol_covs)
      columns += [f"hol_{i}" for i in range(len(HOLIDAYS))]

    return pd.DataFrame(
        data=np.vstack(all_covs).transpose(),
        columns=columns,
        index=self.dti,
    )