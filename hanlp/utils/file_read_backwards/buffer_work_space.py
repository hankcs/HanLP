#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""BufferWorkSpace module."""

import os

new_lines = ["\r\n", "\n", "\r"]
new_lines_bytes = [n.encode("ascii") for n in new_lines]  # we only support encodings that's backward compat with ascii


class BufferWorkSpace:

    """It is a helper module for FileReadBackwards."""

    def __init__(self, fp, chunk_size):
        """Convention for the data.

        When read_buffer is not None, it represents contents of the file from `read_position` onwards
            that has not been processed/returned.
        read_position represents the file pointer position that has been read into read_buffer
            initialized to be just past the end of file.
        """
        self.fp = fp
        self.read_position = _get_file_size(self.fp)  # set the previously read position to the
        self.read_buffer = None
        self.chunk_size = chunk_size

    def add_to_buffer(self, content, read_position):
        """Add additional bytes content as read from the read_position.

        Args:
            content (bytes): data to be added to buffer working BufferWorkSpac.
            read_position (int): where in the file pointer the data was read from.
        """
        self.read_position = read_position
        if self.read_buffer is None:
            self.read_buffer = content
        else:
            self.read_buffer = content + self.read_buffer

    def yieldable(self):
        """Return True if there is a line that the buffer can return, False otherwise."""
        if self.read_buffer is None:
            return False

        t = _remove_trailing_new_line(self.read_buffer)
        n = _find_furthest_new_line(t)
        if n >= 0:
            return True

        # we have read in entire file and have some unprocessed lines
        if self.read_position == 0 and self.read_buffer is not None:
            return True
        return False

    def return_line(self):
        """Return a new line if it is available.

        Precondition: self.yieldable() must be True
        """
        assert(self.yieldable())

        t = _remove_trailing_new_line(self.read_buffer)
        i = _find_furthest_new_line(t)

        if i >= 0:
            l = i + 1
            after_new_line = slice(l, None)
            up_to_include_new_line = slice(0, l)
            r = t[after_new_line]
            self.read_buffer = t[up_to_include_new_line]
        else:  # the case where we have read in entire file and at the "last" line
            r = t
            self.read_buffer = None
        return r

    def read_until_yieldable(self):
        """Read in additional chunks until it is yieldable."""
        while not self.yieldable():
            read_content, read_position = _get_next_chunk(self.fp, self.read_position, self.chunk_size)
            self.add_to_buffer(read_content, read_position)

    def has_returned_every_line(self):
        """Return True if every single line in the file has been returned, False otherwise."""
        if self.read_position == 0 and self.read_buffer is None:
            return True
        return False


def _get_file_size(fp):
    return os.fstat(fp.fileno()).st_size


def _get_next_chunk(fp, previously_read_position, chunk_size):
    """Return next chunk of data that we would from the file pointer.

    Args:
        fp: file-like object
        previously_read_position: file pointer position that we have read from
        chunk_size: desired read chunk_size

    Returns:
        (bytestring, int): data that has been read in, the file pointer position where the data has been read from
    """
    seek_position, read_size = _get_what_to_read_next(fp, previously_read_position, chunk_size)
    fp.seek(seek_position)
    read_content = fp.read(read_size)
    read_position = seek_position
    return read_content, read_position


def _get_what_to_read_next(fp, previously_read_position, chunk_size):
    """Return information on which file pointer position to read from and how many bytes.

    Args:
        fp
        past_read_positon (int): The file pointer position that has been read previously
        chunk_size(int): ideal io chunk_size

    Returns:
        (int, int): The next seek position, how many bytes to read next
    """
    seek_position = max(previously_read_position - chunk_size, 0)
    read_size = chunk_size

    # examples: say, our new_lines are potentially "\r\n", "\n", "\r"
    # find a reading point where it is not "\n", rewind further if necessary
    # if we have "\r\n" and we read in "\n",
    # the next iteration would treat "\r" as a different new line.
    # Q: why don't I just check if it is b"\n", but use a function ?
    # A: so that we can potentially expand this into generic sets of separators, later on.
    while seek_position > 0:
        fp.seek(seek_position)
        if _is_partially_read_new_line(fp.read(1)):
            seek_position -= 1
            read_size += 1  # as we rewind further, let's make sure we read more to compensate
        else:
            break

    # take care of special case when we are back to the beginnin of the file
    read_size = min(previously_read_position - seek_position, read_size)
    return seek_position, read_size


def _remove_trailing_new_line(l):
    """Remove a single instance of new line at the end of l if it exists.

    Returns:
        bytestring
    """
    # replace only 1 instance of newline
    # match longest line first (hence the reverse=True), we want to match "\r\n" rather than "\n" if we can
    for n in sorted(new_lines_bytes, key=lambda x: len(x), reverse=True):
        if l.endswith(n):
            remove_new_line = slice(None, -len(n))
            return l[remove_new_line]
    return l


def _find_furthest_new_line(read_buffer):
    """Return -1 if read_buffer does not contain new line otherwise the position of the rightmost newline.

    Args:
        read_buffer (bytestring)

    Returns:
        int: The right most position of new line character in read_buffer if found, else -1
    """
    new_line_positions = [read_buffer.rfind(n) for n in new_lines_bytes]
    return max(new_line_positions)


def _is_partially_read_new_line(b):
    """Return True when b is part of a new line separator found at index >= 1, False otherwise.

    Args:
        b (bytestring)

    Returns:
        bool
    """
    for n in new_lines_bytes:
        if n.find(b) >= 1:
            return True
    return False
