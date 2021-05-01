# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-24 22:12
import datetime
import io
import logging
import os
import sys
from logging import LogRecord

import termcolor


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%', enable=True):
        super().__init__(fmt, datefmt, style)
        self.enable = enable

    def formatMessage(self, record: LogRecord) -> str:
        message = super().formatMessage(record)
        if self.enable:
            return color_format(message)
        else:
            return remove_color_tag(message)


def init_logger(name=None, root_dir=None, level=logging.INFO, mode='w',
                fmt="%(asctime)s %(levelname)s %(message)s",
                datefmt='%Y-%m-%d %H:%M:%S') -> logging.Logger:
    if not name:
        name = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    rootLogger = logging.getLogger(os.path.join(root_dir, name) if root_dir else name)
    rootLogger.propagate = False

    consoleHandler = logging.StreamHandler(sys.stdout)  # stderr will be rendered as red which is bad
    consoleHandler.setFormatter(ColoredFormatter(fmt, datefmt=datefmt))
    attached_to_std = False
    for handler in rootLogger.handlers:
        if isinstance(handler, logging.StreamHandler):
            if handler.stream == sys.stderr or handler.stream == sys.stdout:
                attached_to_std = True
                break
    if not attached_to_std:
        rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(level)
    consoleHandler.setLevel(level)

    if root_dir:
        os.makedirs(root_dir, exist_ok=True)
        log_path = "{0}/{1}.log".format(root_dir, name)
        fileHandler = logging.FileHandler(log_path, mode=mode)
        fileHandler.setFormatter(ColoredFormatter(fmt, datefmt=datefmt, enable=False))
        rootLogger.addHandler(fileHandler)
        fileHandler.setLevel(level)

    return rootLogger


logger = init_logger(name='hanlp', level=os.environ.get('HANLP_LOG_LEVEL', 'INFO'))


def enable_debug(debug=True):
    logger.setLevel(logging.DEBUG if debug else logging.ERROR)


class ErasablePrinter(object):
    def __init__(self, out=sys.stderr):
        self._last_print_width = 0
        self.out = out

    def erase(self):
        if self._last_print_width:
            self.out.write("\b" * self._last_print_width)
            self.out.write(" " * self._last_print_width)
            self.out.write("\b" * self._last_print_width)
            self.out.write("\r")  # \r is essential when multi-lines were printed
            self._last_print_width = 0

    def print(self, msg: str, color=True):
        self.erase()
        if color:
            msg, _len = color_format_len(msg)
            self._last_print_width = _len
        else:
            self._last_print_width = len(msg)
        self.out.write(msg)
        self.out.flush()


_printer = ErasablePrinter()


def flash(line: str, color=True):
    _printer.print(line, color)


def color_format(msg: str):
    for tag in termcolor.COLORS, termcolor.HIGHLIGHTS, termcolor.ATTRIBUTES:
        for c, v in tag.items():
            start, end = f'[{c}]', f'[/{c}]'
            msg = msg.replace(start, '\033[%dm' % v).replace(end, termcolor.RESET)
    return msg


def remove_color_tag(msg: str):
    for tag in termcolor.COLORS, termcolor.HIGHLIGHTS, termcolor.ATTRIBUTES:
        for c, v in tag.items():
            start, end = f'[{c}]', f'[/{c}]'
            msg = msg.replace(start, '').replace(end, '')
    return msg


def color_format_len(msg: str):
    _len = len(msg)
    for tag in termcolor.COLORS, termcolor.HIGHLIGHTS, termcolor.ATTRIBUTES:
        for c, v in tag.items():
            start, end = f'[{c}]', f'[/{c}]'
            msg, delta = _replace_color_offset(msg, start, '\033[%dm' % v)
            _len -= delta
            msg, delta = _replace_color_offset(msg, end, termcolor.RESET)
            _len -= delta
    return msg, _len


def _replace_color_offset(msg: str, color: str, ctrl: str):
    chunks = msg.split(color)
    delta = (len(chunks) - 1) * len(color)
    return ctrl.join(chunks), delta


def cprint(*args, file=None, **kwargs):
    out = io.StringIO()
    print(*args, file=out, **kwargs)
    text = out.getvalue()
    out.close()
    c_text = color_format(text)
    print(c_text, end='', file=file)


def main():
    # cprint('[blink][yellow]...[/yellow][/blink]')
    # show_colors_and_formats()
    show_colors()
    # print('previous', end='')
    # for i in range(10):
    #     flash(f'[red]{i}[/red]')


def show_colors_and_formats():
    msg = ''
    for c in termcolor.COLORS.keys():
        for h in termcolor.HIGHLIGHTS.keys():
            for a in termcolor.ATTRIBUTES.keys():
                msg += f'[{c}][{h}][{a}] {c}+{h}+{a} [/{a}][/{h}][/{c}]'
    logger.info(msg)


def show_colors():
    msg = ''
    for c in termcolor.COLORS.keys():
        cprint(f'[{c}]"{c}",[/{c}]')


# Generates tables for Doxygen flavored Markdown.  See the Doxygen
# documentation for details:
#   http://www.doxygen.nl/manual/markdown.html#md_tables

# Translation dictionaries for table alignment


if __name__ == '__main__':
    main()
