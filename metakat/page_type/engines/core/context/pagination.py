"""Frozen training pagination policy: noisy unbracketed candidates, never inferred brackets."""
import math
import re

import numpy as np

RECORD = np.dtype([('value', '<u4'), ('system', 'u1')])
POLICY = 'db_candidate_single_v1'
ROMAN = re.compile(r'M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})', re.I)


def candidate(text):
    """Single Arabic/Roman value, or explicitly printed part of `55 [58]`.

    Unbracketed DB values are noisy candidates, not proof of visible printing.
    Multiple numbers, ranges, suffixes and malformed brackets are unavailable.
    """
    text = text.strip()
    correction = re.fullmatch(r'([^\[\]]+)\s+\[[^\[\]]+\]', text)
    if correction:
        text = correction[1].strip()
    if re.fullmatch(r'[0-9]{1,9}', text):
        value = int(text)
        return (value, 1) if value > 0 else (0, 0)
    if text and ROMAN.fullmatch(text):
        digits = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        values = [digits[c] for c in text.upper()]
        return sum(-a if a < b else a for a, b in zip(values, values[1:] + [0])), 2
    return 0, 0


def encode(records, cap=10000):
    """[log magnitude, odd parity, Arabic, Roman, availability], all in [0,1]."""
    if cap <= 1:
        raise ValueError('pagination cap must exceed one')
    value, system = records['value'], records['system']
    valid = (value > 0) & np.isin(system, [1, 2])
    result = np.zeros((len(records), 5), np.float32)
    result[:, 0] = np.log1p(np.minimum(value, cap).astype(np.float64)) / math.log1p(cap)
    result[:, 1] = value % 2
    result[:, 2] = system == 1
    result[:, 3] = system == 2
    result[:, 4] = valid
    result[~valid] = 0
    return result.astype(np.float16)


def encode_page_numbers(numbers, cap=10000):
    """Apply the training parser and FP16 rounding to ordered optional page numbers."""
    records = np.array([candidate(number or '') for number in numbers], RECORD)
    return encode(records, cap).astype(np.float32)
