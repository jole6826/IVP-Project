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

def fromVecToRGB(rgb_vec, shape, order = 'bgr'):
    '''
    Function that converts a vector containing a rgb image in the form rrrgggbbb values to
    a MxNxC matrix where M,N,C are stored in shape = [M, N, C]
    '''
    rgbRows = np.reshape(rgb_vec, [shape[2], shape[0] * shape[1]])
    r_flat = rgbRows[0, :]
    g_flat = rgbRows[1, :]
    b_flat = rgbRows[2, :]
    rgb = np.zeros((shape[0], shape[1], 3,), dtype=np.uint8)
    if order == 'bgr':
        # for open cv
        rgb[:, :, 0] = np.reshape(b_flat, [shape[0], shape[1]])
        rgb[:, :, 1] = np.reshape(g_flat, [shape[0], shape[1]])
        rgb[:, :, 2] = np.reshape(r_flat, [shape[0], shape[1]])
    elif order == 'rgb':
        # for plotting with pyplot
        rgb[:, :, 2] = np.reshape(b_flat, [shape[0], shape[1]])
        rgb[:, :, 1] = np.reshape(g_flat, [shape[0], shape[1]])
        rgb[:, :, 0] = np.reshape(r_flat, [shape[0], shape[1]])
    else:
        print('Format of order is not known, must be rgb (pyplot) or bgr (openCV)')
        return 2
    return rgb


def decodeWithHuffman(file):
    file = file + '_hm.bin'
    img_data, cb, shape = readHuffmanBitstream(file)
    rgbFlat_decoded = hc.huffmanDecoder(img_data, cb)
    rgb = fromVecToRGB(rgbFlat_decoded,shape,order='bgr')

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

    rgb = fromVecToRGB(img_vec,[height,width,nChannels], order='bgr')

    return rgb.astype(np.uint8)
