# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-27 00:01
import datetime
import time


def human_time_delta(days, hours, minutes, seconds, delimiter=' ') -> str:
    units = locals().copy()
    units.pop('delimiter')
    non_zero = False
    result = []
    for key, val in sorted(units.items()):
        append = False
        if non_zero:
            append = True
        elif val:
            non_zero = True
            append = True
        if append:
            result.append('{} {}'.format(val, key[0]))
    if not non_zero:
        return '0 s'
    return delimiter.join(result)


def seconds_to_time_delta(seconds):
    seconds = round(seconds)
    days = seconds // 86400
    hours = seconds // 3600 % 24
    minutes = seconds // 60 % 60
    seconds = seconds % 60
    return days, hours, minutes, seconds


def report_time_delta(seconds, human=True):
    days, hours, minutes, seconds = seconds_to_time_delta(seconds)
    if human:
        return human_time_delta(days, hours, minutes, seconds)
    return days, hours, minutes, seconds


class HumanTimeDelta(object):

    def __init__(self, delta_seconds) -> None:
        super().__init__()
        self.delta_seconds = delta_seconds

    def report(self, human=True):
        return report_time_delta(self.delta_seconds, human)

    def __str__(self) -> str:
        return self.report(human=True)

    def __truediv__(self, scalar):
        return HumanTimeDelta(self.delta_seconds / scalar)


class Timer(object):
    def __init__(self) -> None:
        self.last = time.time()

    def start(self):
        self.last = time.time()

    def stop(self) -> HumanTimeDelta:
        now = time.time()
        seconds = now - self.last
        self.last = now
        return HumanTimeDelta(seconds)


def now_human(year='y'):
    now = datetime.datetime.now()
    return now.strftime(f"%{year}-%m-%d %H:%M:%S")


def now_datetime():
    return now_human('Y')


def now_filename(fmt="%y%m%d_%H%M%S"):
    """
    Generate filename using current datetime, in 20180102_030405 format
    Returns
    -------

    """
    now = datetime.datetime.now()
    return now.strftime(fmt)
