#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 11:19:49 2021

https://github.com/pytorch/pytorch/issues/13246#issuecomment-617140519
"""
import numpy as np
import torch
from typing import Union

# --- UTILITY FUNCTIONS ---

def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)

def sequence_to_string(seq: np.ndarray) -> str:
    return ''.join([chr(c) for c in seq])

def pack_sequences(seqs: Union[np.ndarray, list]) -> (np.ndarray, np.ndarray):
    values = np.concatenate(seqs, axis=0)
    offsets = np.cumsum([len(s) for s in seqs])
    return values, offsets

def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    off1 = offsets[index]
    if index > 0:
        off0 = offsets[index - 1]
    elif index == 0:
        off0 = 0
    else:
        raise ValueError(index)
    return values[off0:off1]


def pack(strings:list):
    return pack_sequences( [string_to_sequence(s) for s in strings] )

def unpack(pack, idx):
    return sequence_to_string(unpack_sequence(pack[0], pack[1], idx))