import numpy as np
import os
import huffmanCoding as hc
import bit_handling as bh
import pickle


def decodeWithHuffman(file):
    file = file + '.bin'
    img_data, cb, shape = readHuffmanBitstream(file)
    rgbFlat_decoded = hc.huffmanDecoder(img_data, cb)
    rgbRows = np.reshape(rgbFlat_decoded, [shape[2], shape[0] * shape[1]])
    r_flat = rgbRows[0, :]
    g_flat = rgbRows[1, :]
    b_flat = rgbRows[2, :]
    rgb = np.zeros((shape[0], shape[1], 3,), dtype=np.uint8)
    rgb[:, :, 0] = np.reshape(b_flat, [shape[0], shape[1]])
    rgb[:, :, 1] = np.reshape(g_flat, [shape[0], shape[1]])
    rgb[:, :, 2] = np.reshape(r_flat, [shape[0], shape[1]])

    return rgb


def readHuffmanBitstream(file):
    with open(file, 'rb') as f:
        cb = pickle.load(f)
        shape = pickle.load(f)
        bin_data = pickle.load(f)

    data = bh.unpack_bytes_to_bits(bin_data)
    lenData = len(data)
    n_padded_zeros = int(data[0:3], 2)
    if n_padded_zeros == 0:
        img_data = data[3:]
    else:
        img_data = data[3:-n_padded_zeros]

    return img_data, cb, shape