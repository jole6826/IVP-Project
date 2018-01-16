import numpy as np
import os
import bit_handling as bh
import huffmanCoding as hc
import pickle
import helpers as help

def print_usage():
    print("Ohps, something went wrong. Check parameters")



def encodeWithHuffman(img,outPath,dumpHuffman = True, dumpRGB = False):
    # img       - is the image that shall be encoded using Huffman
    #           should be 1 or 3 channel image matrix
    # outPath   - is the path where it should be saved to
    #            will be saved as outPath.bin
    rgbFlat, height, width, nChannels = help.flattenImg(img)

    codebook, codebook_tree = hc.createHuffmanCodebook(rgbFlat)
    data_bits = hc.huffmanEncoder(rgbFlat, codebook)

    padded_bits = bh.pad_bits(data_bits)
    data_binstring = bh.pack_bits_to_bytes(padded_bits)

    if dumpHuffman:
        dump_fname = outPath + "_hm.bin"
        with open(os.path.join(dump_fname), 'wb') as f:
            pickle.dump(codebook, f, 1)
            pickle.dump([height, width, nChannels], f, 1)
            pickle.dump(data_binstring, f, 1)

    if dumpRGB:
        dump_fname = outPath + "_rgb.bin"
        with open(os.path.join(dump_fname), 'wb') as f:
            pickle.dump([height, width, nChannels], f, 1)
            pickle.dump(rgbFlat, f, 1)

    return 0

def encodeWithRunLength(img, outPath, dumpRunLength = True, dumpRGB = False):
    # img       - is the image that shall be encoded using Huffman
    #           should be 1 or 3 channel image matrix
    # outPath   - is the path where it should be saved to
    #            will be saved as outPath.bin

    rgbFlat, height, width, nChannels = help.flattenImg(img)

    rgbFlat = np.hstack((rgbFlat, rgbFlat[-1]+1))
    ### run length ######################################################################
    prevValue = rgbFlat[0]

    count = 1
    encoded = []
    for currValue in rgbFlat[1::]:
        if prevValue == currValue:
            count += 1
        else:
            encoded.append([prevValue, count])
            prevValue = currValue
            count = 1

    encoded_vals = np.array(encoded)[:,0].astype(np.uint8)
    encoded_counts = np.array(encoded)[:,1]

    if max(encoded_counts) < 255:
        encoded_counts = encoded_counts.astype(np.uint8)
    elif max(encoded_counts) < 65535:
        encoded_counts = encoded_counts.astype(np.uint16)
    else:
        encoded_counts = encoded_counts.astype(np.uint32)

    if dumpRunLength:
        dump_fname = outPath + "_rl.bin"
        with open(os.path.join(dump_fname), 'wb') as f:
            pickle.dump([height, width, nChannels], f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(encoded_vals, f, 1)
            pickle.dump(encoded_counts, f, 1)

    if dumpRGB:
        dump_fname = outPath + "_rgb.bin"
        with open(os.path.join(dump_fname), 'wb') as f:
            pickle.dump([height, width, nChannels], f, 1)
            pickle.dump(rgbFlat, f, 1)

    return 0


