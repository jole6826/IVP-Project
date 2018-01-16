import numpy as np
import os
import huffmanCoding as hc
import bit_handling as bh
import pickle
import helpers as help


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


def decodeWithHuffman(file):
    file = file + '_hm.bin'
    img_data, cb, shape = readHuffmanBitstream(file)
    rgbFlat_decoded = hc.huffmanDecoder(img_data, cb)
    rgb = help.fromVecToRGB(rgbFlat_decoded,shape,order='bgr')

    return rgb

def decodeWithRunLegth(file):

    # loading data from .bin file
    file = file + '_rl.bin'
    with open(file, 'rb') as f:
        [height, width, nChannels] = pickle.load(f)
        rl_vals = pickle.load(f)
        rl_counts = pickle.load(f)

    rl_data = np.vstack((rl_vals, rl_counts))

    # reconstruction of original rgb image
    img_vec = np.zeros([1,height*width*nChannels])
    ix = 0
    for val, nElements in rl_data[::, ::].T:
        n = nElements
        img_vec[:,ix:ix+n] = val
        ix += n

    rgb = help.fromVecToRGB(img_vec,[height,width,nChannels], order='bgr')

    return rgb.astype(np.uint8)
