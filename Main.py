import struct
import numpy as np
import os
import cv2
import lib.huffmanCoding as hc
import pickle


dumpHuffman = True
dumpRGB = True
pathToImgs = "/home/jole/Dokumente/Studium/Master/1_Semester_WS1718/IVP/Project-Excercise/Code/imgs"
wallaby = "wallaby.png"
lena = "lena512color.tiff"
moon = 'MoonImage.tif'

imgPath = os.path.join(pathToImgs,moon)
img = cv2.imread(imgPath)
shape = img.shape
if len(shape) == 3:
    maxHeight = shape[0]
    maxWidth = shape[1]
    nChannels = shape[2]
else:
    maxHeight = shape[0]
    maxWidth = shape[1]
    nChannels = 1

# Crop for wallaby
# loR = 300
# loC = 700
# width = 1024

# Crop for lena and moon (no crop)
# loR = 0
# loC = 0
# width = 512
# height = width

# Crop for moon
loR = 0
loC = 0
width = maxWidth
height = maxHeight
assert (loR <= maxWidth and loC <= maxHeight and (loR+height) <= maxHeight and (loC+width) <= maxWidth)
img = img[loR:loR+height,loC:loC+width,:]


bFlat = img[:,:,0].flatten()
gFlat = img[:,:,1].flatten()
rFlat = img[:,:,2].flatten()
rgbFlat = np.hstack((rFlat,gFlat,bFlat))

codebook, codebook_tree = hc.createHuffmanCodebook(rgbFlat)
data_bits = hc.huffmanEncoder(rgbFlat,codebook)

cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


n_zero_pad = 8 - ((len(data_bits)+3) % 8)
if n_zero_pad == 8:
    n_zero_pad = 0
    padded_bits = '{:b}'.format(n_zero_pad).zfill(3) + data_bits
else:
    padded_bits = '{:b}'.format(n_zero_pad).zfill(3) + data_bits + '{:b}'.format(0).zfill(n_zero_pad)

data_binstring = hc.pack_bits_to_bytes(padded_bits)

if dumpHuffman:
    dump_fname = 'rgb_huffman.bin'
    with open(os.path.join('bin', dump_fname), 'wb') as f:
        pickle.dump(codebook, f, 1)
        pickle.dump([height, width, nChannels], f, 1)
        pickle.dump(data_binstring, f, 1)
if dumpRGB:
    dump_fname = 'rgb.bin'
    with open(os.path.join('bin', dump_fname), 'wb') as f:
        pickle.dump(rgbFlat, f, 1)




