import numpy as np
import os
import bit_handling as bh
import huffmanCoding as hc
import pickle

def print_usage():
    print "Ohps, something went wrong. Check parameters"


def flattenImg(img):
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

    return rgbFlat, height, width, nChannels

def encodeWithHuffman(img,outPath,dumpHuffman = True, dumpRGB = False):
    # img       - is the image that shall be encoded using Huffman
    #           should be 1 or 3 channel image matrix
    # outPath   - is the path where it should be saved to
    #            will be saved as outPath.bin
    rgbFlat, height, width, nChannels = flattenImg(img)

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

    rgbFlat, height, width, nChannels = flattenImg(img)

    rgbFlat = np.hstack((rgbFlat, rgbFlat[-1]+1))
    ### run length ######################################################################
    value = rgbFlat[0]
    prevValue = rgbFlat[0]
    # ! BEACHTEN: count = Anzahl der Wiederholungen
    #   beim Wiederherstellen -> Anzahl des Wertes = count + 1
    count = 1
    encoded = []
    for currValue in rgbFlat[1::]:
        if prevValue == currValue:
            count += 1
        else:
            encoded.append([prevValue, count])
            prevValue = currValue
            count = 1

    # for index in range(0, len(rgbFlat)+1):
    #     if value == rgbFlat[index]:
    #         count += 1
    #     else:
    #         # append = anhaengen
    #         encoded.append([value, count])
    #         value = rgbFlat[index]
    #         count = 1



    encoded_vec = np.array(encoded).reshape(len(encoded) * 2)
    encoded_vec = encoded_vec.astype(np.uint16)
    encoded_vals = np.array(encoded)[:,0].astype(np.uint8)
    encoded_counts = np.array(encoded)[:,1].astype(np.uint16)
    if dumpRunLength:
        dump_fname = outPath + "_rl.bin"
        with open(os.path.join(dump_fname), 'wb') as f:
            pickle.dump([height, width, nChannels], f, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump(encoded_vec, f, 1)
            pickle.dump(encoded_vals, f, 1)
            pickle.dump(encoded_counts, f, 1)

    if dumpRGB:
        dump_fname = outPath + "_rgb.bin"
        with open(os.path.join(dump_fname), 'wb') as f:
            pickle.dump([height, width, nChannels], f, 1)
            pickle.dump(rgbFlat, f, 1)

    return 0


