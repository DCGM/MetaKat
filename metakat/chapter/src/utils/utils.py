# File containing simple utility functions.
# Author: Richard Blažo
# File name: utils.py
# Description: This file contains simple utility functions for debugging and other purposes.

_debug = False


def set_debug(debug):
    global _debug
    _debug = debug


def debugprint(*args):
    if _debug:
        print(*args)
