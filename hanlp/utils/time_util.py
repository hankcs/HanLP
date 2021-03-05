# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-27 00:01
import datetime
import logging
import sys
import time
from typing import Union

from hanlp.utils.log_util import ErasablePrinter, color_format, color_format_len


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


class CountdownTimer(ErasablePrinter):

    def __init__(self, total: int, out=sys.stdout) -> None:
        super().__init__(out=out)
        self.total = total
        self.current = 0
        self.start = time.time()
        self.finished_in = None
        self.last_log_time = 0

    def update(self, n=1):
        self.current += n
        self.current = min(self.total, self.current)
        if self.current == self.total:
            self.finished_in = time.time() - self.start

    @property
    def ratio(self) -> str:
        return f'{self.current}/{self.total}'

    @property
    def ratio_percentage(self) -> str:
        return f'{self.current / self.total:.2%}'

    @property
    def eta(self) -> float:
        elapsed = self.elapsed
        if self.finished_in:
            eta = 0
        else:
            eta = elapsed / max(self.current, 0.1) * (self.total - self.current)

        return eta

    @property
    def elapsed(self) -> float:
        if self.finished_in:
            elapsed = self.finished_in
        else:
            elapsed = time.time() - self.start
        return elapsed

    @property
    def elapsed_human(self) -> str:
        return human_time_delta(*seconds_to_time_delta(self.elapsed))

    @property
    def elapsed_average(self) -> float:
        return self.elapsed / self.current

    @property
    def elapsed_average_human(self) -> str:
        return human_time_delta(*seconds_to_time_delta(self.elapsed_average))

    @property
    def eta_human(self) -> str:
        return human_time_delta(*seconds_to_time_delta(self.eta))

    @property
    def total_time(self) -> float:
        elapsed = self.elapsed
        if self.finished_in:
            t = self.finished_in
        else:
            t = elapsed / max(self.current, 1) * self.total

        return t

    @property
    def total_time_human(self) -> str:
        return human_time_delta(*seconds_to_time_delta(self.total_time))

    def stop(self, total=None):
        if not self.finished_in or total:
            self.finished_in = time.time() - self.start
            if not total:
                self.total = self.current
            else:
                self.current = total
                self.total = total

    @property
    def et_eta(self):
        _ = self.elapsed
        if self.finished_in:
            return self.elapsed
        else:
            return self.eta

    @property
    def et_eta_human(self):
        text = human_time_delta(*seconds_to_time_delta(self.et_eta))
        if self.finished_in:
            return f'ET: {text}'
        else:
            return f'ETA: {text}'

    @property
    def finished(self):
        return self.total == self.current

    def log(self, info=None, ratio_percentage=True, ratio=True, step=1, interval=0.5, erase=True,
            logger: Union[logging.Logger, bool] = None, newline=False, ratio_width=None):
        self.update(step)
        now = time.time()
        if now - self.last_log_time > interval or self.finished:
            cells = []
            if ratio_percentage:
                cells.append(self.ratio_percentage)
            if ratio:
                ratio = self.ratio
                if not ratio_width:
                    ratio_width = self.ratio_width
                ratio = ratio.rjust(ratio_width)
                cells.append(ratio)
            cells += [info, self.et_eta_human]
            cells = [x for x in cells if x]
            msg = f'{" ".join(cells)}'
            self.last_log_time = now
            self.print(msg, newline, erase, logger)

    @property
    def ratio_width(self) -> int:
        return len(f'{self.total}') * 2 + 1

    def print(self, msg, newline=False, erase=True, logger=None):
        self.erase()
        msg_len = 0 if newline else len(msg)
        if self.finished and logger:
            sys.stdout.flush()
            if isinstance(logger, logging.Logger):
                logger.info(msg)
        else:
            msg, msg_len = color_format_len(msg)
            sys.stdout.write(msg)
            if newline:
                sys.stdout.write('\n')
                msg_len = 0
        self._last_print_width = msg_len
        if self.finished and not logger:
            if erase:
                self.erase()
            else:
                sys.stdout.write("\n")
                self._last_print_width = 0
        sys.stdout.flush()


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
    """Generate filename using current datetime, in 20180102_030405 format

    Args:
      fmt:  (Default value = "%y%m%d_%H%M%S")

    Returns:

    
    """
    now = datetime.datetime.now()
    return now.strftime(fmt)
