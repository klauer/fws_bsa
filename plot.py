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
    @property
    def value(self):
        # all entries are using only the lower 32-bits
        return self.data[2]


class UIntEntry(EntryBase):
    _fields_ = [('data', ctypes.c_uint32 * 3)]


class IntEntry(EntryBase):
    _fields_ = [('data', ctypes.c_int32 * 3)]


class FloatEntry(EntryBase):
    _fields_ = [('data', ctypes.c_float * 3)]


class Entry(ctypes.LittleEndianStructure):
    _fields_ = [('data', ctypes.c_uint32 * 3)]
    repr_type = 'raw'

    def as_pulse_id(self):
        return (self.data[2] << 32) | self.data[1]

    def as_raw(self):
        return ((self.data[1] << 16) |
                (self.data[0] >> 16)
                )

    def as_uint96(self):
        return ((self.data[2] << 64) |
                (self.data[1] << 32) |
                (self.data[0])
                )

    def __repr__(self):
        if self.repr_type == 'pulse_id':
            return repr(self.as_pulse_id())
        elif self.repr_type == 'float':
            return repr(self.as_float())
        elif self.repr_type == 'raw':
            return repr(self.as_raw())
        else:
            return repr(self.as_uint96())


class WireBSAData(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [('header', Header),
                ('speed', UIntEntry),
                ('lower_limit', UIntEntry),
                ('signed_speed', IntEntry),
                ('position', UIntEntry),
                ('pmt_integer', UIntEntry),
                ('pmt_float', FloatEntry),
                ('status', UIntEntry),
                ('remaining_entries', Entry * (31 - 7)),
                ]

    named_entries = ['speed',
                     'lower_limit',
                     'signed_speed',
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

            info['nchan'].append(data.header.nchan)
            header = data.header
            for name in cls.named_entries:
                info[name].append(getattr(data, name).value)
            info['pulse_id_lower'].append(header.pulse_id_lower)
            info['pulse_id_upper'].append(header.pulse_id_upper)
            info['pulse_id'].append((header.pulse_id_upper << 32) |
                                    header.pulse_id_lower)

    return info


def sign_extend(value, *, bits=18):
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


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
    # plt.plot(np.asarray(data['speed']))
    plt.plot([sign_extend(speed, bits=18) for speed in data['speed']])
    plt.title('Speed')
    plt.show()


if __name__ == '__main__':
    plot(sys.argv[1])
