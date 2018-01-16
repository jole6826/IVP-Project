import numpy as np
import matplotlib.pyplot as plt

def entropy(img, showPlot=False):
    # calculate entropy of error images (what has to be transmitted)

    maxVal = max(img.flatten())
    minVal = min(img.flatten())
    nBins = np.int16(maxVal - minVal + 1)
    hist, bins = np.histogram(img.flatten(), bins=nBins)
    probability = np.float32(hist) / (img.size)
    logTerm = np.log2(probability[hist != 0])
    ent = -np.dot(probability[hist != 0], logTerm)

    if showPlot:
        plt.plot(bins[0:nBins], np.transpose(hist))
        plt.show()
        plt.plot(bins[0:nBins], probability)
        plt.show()

    return ent


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
