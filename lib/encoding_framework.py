import numpy as np
import os
import bit_handling as bh
import huffmanCoding as hc
import pickle

def print_usage():
    print "Ohps, something went wrong. Check parameters"

def encodeWithHuffman(img,outPath,dumpHuffman = True, dumpRGB = False):
    # img       - is the image that shall be encoded using Huffman
    #           should be 1 or 3 channel image matrix
    # outPath   - is the path where it should be saved to
    #            will be saved as outPath.bin
    shape = img.shape

    if len(shape) == 3:
        bFlat = img[:, :, 0].flatten()
        gFlat = img[:, :, 1].flatten()
        rFlat = img[:, :, 2].flatten()
        rgbFlat = np.hstack((rFlat, gFlat, bFlat))
        height = shape[0]
        width = shape[1]
        nChannels = shape[2]
    elif len(shape) == 1:
        rgbFlat = img.flatten()
        height = shape[0]
        width = shape[1]
        nChannels = 1
    else:
        print_usage()
        return 2

    codebook, codebook_tree = hc.createHuffmanCodebook(rgbFlat)
    data_bits = hc.huffmanEncoder(rgbFlat, codebook)

    padded_bits = bh.pad_bits(data_bits)
    data_binstring = bh.pack_bits_to_bytes(padded_bits)

    dump_fname = outPath + ".bin"
    if dumpHuffman:
        with open(os.path.join(dump_fname), 'wb') as f:
            pickle.dump(codebook, f, 1)
            pickle.dump([height, width, nChannels], f, 1)
            pickle.dump(data_binstring, f, 1)

    if dumpRGB:
        dump_fname = outPath + "_rgb.bin"
        with open(os.path.join(dump_fname), 'wb') as f:
            pickle.dump(rgbFlat, f, 1)

    return 0


