import numpy as np
import os
import huffmanCoding as hc
import bit_handling as bh
import pickle


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
    rgbRows = np.reshape(rgbFlat_decoded, [shape[2], shape[0] * shape[1]])
    r_flat = rgbRows[0, :]
    g_flat = rgbRows[1, :]
    b_flat = rgbRows[2, :]
    rgb = np.zeros((shape[0], shape[1], 3,), dtype=np.uint8)
    rgb[:, :, 0] = np.reshape(b_flat, [shape[0], shape[1]])
    rgb[:, :, 1] = np.reshape(g_flat, [shape[0], shape[1]])
    rgb[:, :, 2] = np.reshape(r_flat, [shape[0], shape[1]])

    return rgb

def decodeWithRunLegth(file):
    file = file + '_rl.bin'
    with open(file, 'rb') as f:
        [height, width, nChannels] = pickle.load(f)
        rl_vec = pickle.load(f)

    rl_data = np.reshape(rl_vec, [len(rl_vec)/2, 2])

    # for loops over rl_data to restore values in img_vec
    # reshape img_vec to real shape before return

    # for ix in xrange(0,len(rl_data)):

    # first block
    val1 = rl_data[0,0]
    n1 = rl_data[0,1]
    img_vec = np.ones([1,n1])*val1

    for val, nElements in rl_data[1::,::]:
        #print ix
        # val = rl_data[ix,0]
        # nElements = rl_data[ix,1] + 1
        n = nElements
        #img_vec.append(np.ones([nElements,1])*val)
        newBlock = np.ones([1,n],dtype=np.uint8)*val
        # if ix == 0:
        #     img_vec = newBlock
        # else:
        #     img_vec = np.hstack((img_vec, newBlock))
        img_vec = np.hstack((img_vec, newBlock))

    rgbRows = np.reshape(img_vec, [nChannels, height * width])
    r_flat = rgbRows[0, :]
    g_flat = rgbRows[1, :]
    b_flat = rgbRows[2, :]
    rgb = np.zeros((height, width, 3,), dtype=np.uint8)
    rgb[:, :, 0] = np.reshape(b_flat, [height, width])
    rgb[:, :, 1] = np.reshape(g_flat, [height, width])
    rgb[:, :, 2] = np.reshape(r_flat, [height, width])
    return rgb.astype(np.uint8)
