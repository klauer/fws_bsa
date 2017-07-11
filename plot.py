'''
Usage: python plot.py [bsa_filename.dat_0]
'''

import sys
import ctypes

from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


class Header(ctypes.LittleEndianStructure):
    _fields_ = [('pad', ctypes.c_uint16),
                ('nchan', ctypes.c_uint16),
                ('pulse_id_lower', ctypes.c_uint32),
                ('pulse_id_upper', ctypes.c_uint32),
                ]


class EntryBase(ctypes.LittleEndianStructure):
    _fields_ = [('data', ctypes.c_uint32 * 3)]

    @property
    def raw_value(self):
        return ((self.data[1] << 16) | self.data[0] >> 16)


def sign_extend(value, *, bits=18):
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


class UIntEntry(EntryBase):
    bits = 18

    @property
    def raw_mask(self):
        return (1 << self.bits) - 1

    @property
    def value(self):
        return self.raw_value & self.raw_mask


class IntEntry(EntryBase):
    bits = 18

    @property
    def value(self):
        return sign_extend(self.raw_value, bits=self.bits)


class FloatEntry(EntryBase):
    @property
    def fixed(self):
        return bool(self.data[0] & (1 << 15))

    @property
    def n(self):
        return self.data[0] & 0x1fff

    @property
    def mean(self):
        if self.fixed:
            return float(self.raw_value)

        exception_condition = (self.data[0] & (1 << 13))
        n = self.n
        if not exception_condition and n:
            raw_val = ((self.data[1] & 0xffff) << 16) | (self.data[0] >> 16)
            return float(raw_val) / n

    @property
    def value(self):
        return self.mean


class UnknownEntry(EntryBase):
    ...


class WireBSAData(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [('header', Header),
                ('speed_freq', UIntEntry),
                ('lower_limit', UIntEntry),
                ('signed_speed_freq', IntEntry),
                ('position', IntEntry),
                ('pmt_integer', UIntEntry),
                ('pmt_float', FloatEntry),
                ('status', UIntEntry),
                ('remaining_entries', UnknownEntry * (31 - 7)),
                ]

    named_entries = ['speed_freq',
                     'lower_limit',
                     'signed_speed_freq',
                     'position',
                     'pmt_integer',
                     'pmt_float',
                     'status',
                     ]


assert ctypes.sizeof(WireBSAData) == 384


def read_file(fn, *, cls=WireBSAData):
    info = defaultdict(lambda: [])
    with open(fn, 'rb') as f:
        f.seek(0, 2)
        data_points = f.tell() // ctypes.sizeof(cls)
        print('Total data points:', data_points)
        f.seek(0, 0)

        data = cls()
        for npoint in range(data_points):
            data.header.nchan = 0
            f.readinto(data)
            assert data.header.nchan == 31

            header = data.header
            for name in cls.named_entries:
                info[name].append(getattr(data, name).value)
            for name in ('nchan', 'pulse_id_lower', 'pulse_id_upper'):
                info['pulse_id_lower'].append(getattr(header, name))
            info['pulse_id'].append((header.pulse_id_upper << 32) |
                                    header.pulse_id_lower)

    return info


def running_mean(x, n):
    return np.convolve(x, np.ones((n, )) / n)[(n - 1):]


def plot(fn):
    print(ctypes.sizeof(WireBSAData), 0x180)
    data = read_file(fn)

    for key, values in sorted(data.items()):
        print(key, values[:10])

    plt.plot(np.asarray(data['position']))
    plt.title('BSA-Acquired Position vs Data Point')
    plt.xlabel('Data point')
    plt.ylabel('Position [um]')

    plt.figure()
    speed_freq = np.asarray(data['speed_freq'])
    # speed in um/s, calculated by number of 156MHz clock pulses between
    # encoder pulses
    speed = 1. / (speed_freq * 6.4e-9)
    # speed = running_mean(speed, 20)
    plt.plot(speed)
    plt.xlabel('Data point')
    plt.ylabel('Speed [um/s]')
    plt.title('BSA-Acquired Speed vs Data Point')
    plt.show()


if __name__ == '__main__':
    plot(sys.argv[1])
