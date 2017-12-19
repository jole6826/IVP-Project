import struct
import numpy as np
import os
import cv2
import lib.huffmanCoding as hc
import pickle



# Read and decode Hufmann coded images

open_fname = 'rgb_huffman.bin'
with open(os.path.join('bin',open_fname),'rb') as f:
    cb = pickle.load(f)
    shape = pickle.load(f)
    bin_data = pickle.load(f)

data = hc.unpack_bytes_to_bits(bin_data)
lenData = len(data)
n_padded_zeros = int(data[0:3], 2)
if n_padded_zeros == 0:
    img_data = data[3:]
else:
    img_data = data[3:-n_padded_zeros]

rgbFlat_decoded = hc.huffmanDecoder(img_data,cb)
rgbRows = np.reshape(rgbFlat_decoded, [shape[2], shape[0]*shape[1]])
r_flat = rgbRows[0,:]
g_flat = rgbRows[1,:]
b_flat = rgbRows[2,:]
rgb = np.zeros((shape[0],shape[1],3,),dtype=np.uint8)
rgb[:,:,0] = np.reshape(b_flat,[shape[0], shape[1]])
rgb[:,:,1] = np.reshape(g_flat,[shape[0], shape[1]])
rgb[:,:,2] = np.reshape(r_flat,[shape[0], shape[1]])

cv2.imshow('Test',rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
test = 1