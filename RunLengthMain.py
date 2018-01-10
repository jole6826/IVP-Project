import glob
import cv2
import numpy as np
import pickle
from os.path import basename

from RunLengthEncoding import runLength

files = glob.glob('imgs/*.jpg')


for file in files:
    filename = basename(files[0])[:-4]
    pic = cv2.imread(file)
    pickle.dump(pic, open(filename+'.p','wb'), protocol=pickle.HIGHEST_PROTOCOL)

    if len(pic.shape) == 2:
        encoded_vec = runLength(pic)
    else:
        b = pic[:,:,0]
        g = pic[:,:,1]
        r = pic[:,:,2]

        encoded_vec_B = runLength(b)
        encoded_vec_G = runLength(g)
        encoded_vec_R = runLength(r)
        encoded_vec = [encoded_vec_B,encoded_vec_G,encoded_vec_R]

    pickle.dump(encoded_vec, open('runLength'+filename+'.p','wb'), protocol=pickle.HIGHEST_PROTOCOL)
