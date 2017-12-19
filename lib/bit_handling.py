import struct
import numpy as np



def pack_bits_to_bytes(bitstream):
    n_bytes = len(bitstream)/8
    bytes_stream = np.zeros(n_bytes, dtype=int)
    for m in range(n_bytes):
        bytes_stream[m] = eval('0b' + bitstream[m*8:(m*8+8)])

    str_bytes = struct.pack('B'*len(bytes_stream), *bytes_stream)
    return str_bytes


def unpack_bytes_to_bits(byte_string):
    unpacked_bytes = struct.unpack('B'*len(byte_string), byte_string)

    bitstream = ''
    for byte in unpacked_bytes:
        #create bit string from byte:
        bits = bin(byte)
        bitstream = bitstream + bits[2:].zfill(8)

    return bitstream

def pad_bits(data_bits):
    n_zero_pad = 8 - ((len(data_bits) + 3) % 8)
    if n_zero_pad == 8:
        n_zero_pad = 0
        padded_bits = '{:b}'.format(n_zero_pad).zfill(3) + data_bits
    else:
        padded_bits = '{:b}'.format(n_zero_pad).zfill(3) + data_bits + '{:b}'.format(0).zfill(n_zero_pad)

        return padded_bits
