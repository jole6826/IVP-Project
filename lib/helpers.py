def entropy(img, showPlot=False):
    # calculate entropy of error images (what has to be transmitted)
    import numpy as np
    import matplotlib.pyplot as plt
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
