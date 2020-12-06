import cv2
import numpy as np
from sklearn.neighbors import KDTree
import sys
import graphcut
import matplotlib.pyplot as plt

PATCH_SIZE = 8
KERNEL_SIZE = 9


def GetOffsets2(patches, indices):
    print("build kdtree")
    kd = KDTree(patches, leaf_size=24)
    #flann = FLANN()
    #kd = scipy.spatial.KDTree(patches[0:10], 16)
    print("query for every patch")
    #offsets = np.zeros((patches.shape[0], 2))
    offsets = [None] * patches.shape[0]
    k = min(1000, patches.shape[0])
    for i in range(patches.shape[0]):
        if (i + 1) % 10000 == 0:
            print(i + 1, "nearest neighbor searched")
        ds, idxs = kd.query(patches[i:i+1], k)
        #idxs, ds = flann.nn(patches.astype(int32), patches[i:i+1].astype(int32), 1000)
        found = False
        for j in range(len(idxs[0])):
            nearest = idxs[0][j]
            #nearest = idxs[i][-1]
            offset = [indices[nearest][0] - indices[i][0], indices[nearest][1] - indices[i][1]]
            if offset[0]**2 + offset[1]**2 >= TAU**2:
                offsets[i] = offset
                found = True
                #print("offset: ", offset)
                break
        if not found:
            nearest = idxs[0][0]
            offsets[i] = [indices[nearest][0] - indices[i][0], indices[nearest][1] - indices[i][1]]
            print("offset not found")
    print("getoffsets2 done")
    return offsets


def GetKDominantOffsets(offsets, K, height, width):
    start = time()
    x, y = [offset[0] for offset in offsets if offset != None], [offset[1] for offset in offsets if offset != None]
    bins = [[i for i in range(np.min(x),np.max(x))], [i for i in range(np.min(y),np.max(y))]]
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)

    p, q = np.where(hist != 0)
    peakOffsets, freq = [[xedges[i], yedges[j]] for (i, j) in zip(p, q)], hist[p, q].flatten()
    peakOffsets = np.array(peakOffsets)
    print(np.array(peakOffsets).shape, np.array(freq).shape)
    plot.ScatterPlot3D(peakOffsets[:,0], peakOffsets[:,1], freq, [height, width])

    hist = hist.T
    hist = cv2.GaussianBlur(hist, KERNEL_SIZE, np.sqrt(2))
    #plot.PlotHistogram2D(hist, xedges, yedges)
    #p, q = np.where(hist == cv2.dilate(hist, np.ones(8))) # Non Maximal Suppression
    nonMaxSuppressedHist = np.zeros(hist.shape)
    #nonMaxSuppressedHist[p, q] = hist[p, q]
    w = 16
    for r in range(0, hist.shape[0], w):
        for c in range(0, hist.shape[1], w):
            r2, c2 = r + w, c + w
            if r2 > hist.shape[0]:
                r2 = hist.shape[0]
            if c2 > hist.shape[1]:
                c2 = hist.shape[1]
            idx = np.argmax(hist[r:r2, c:c2])
            maxr, maxc = idx // (c2 - c), idx % (c2 - c)
            #print("r,r2,c,c2:", r, r2, c, c2)
            #print("idx,maxr,maxc", idx, maxr, maxc)
            nonMaxSuppressedHist[r + maxr, c + maxc] = hist[maxr + r, maxc + c]#np.sum(hist[r:r2, c:c2])#np.sum(hist[r:r2, c:c2] != 0)

    #plot.PlotHistogram2D(nonMaxSuppressedHist, xedges, yedges)
    print(nonMaxSuppressedHist.flatten().shape, hist.shape)
    p, q = np.where(nonMaxSuppressedHist >= np.partition(nonMaxSuppressedHist.flatten(), -K)[-K])
    peakHist = np.zeros(hist.shape)
    peakHist[p, q] = nonMaxSuppressedHist[p, q]
    #plot.PlotHistogram2D(peakHist, xedges, yedges)
    peakOffsets, freq = [[xedges[j], yedges[i]] for (i, j) in zip(p, q)], nonMaxSuppressedHist[p, q].flatten()
    peakOffsets = np.array([x for _, x in sorted(zip(freq, peakOffsets), reverse=True)], dtype="int64")[:2*K]
    end = time()
    print("peakOffsets shape:", peakOffsets.shape, "freq shape: ", freq.shape, "p, q shape:", p.shape, q.shape)
    plot.ScatterPlot3D(peakOffsets[:,0], peakOffsets[:,1], freq, [height, width])
    print("GetKDominantOffsets execution time: ", end - start)
    return peakOffsets


def ScatterPlot3D(x, y, z, domain):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(y, x, z)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_zlabel('frequency')
    ax.set_xlim([-domain[0], domain[0]])
    ax.set_ylim([-domain[1], domain[1]])
    plt.show()


# I consulted with https://github.com/Pranshu258/Image_Completion/
def patch(imagefile, mask):
    # read in image
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(imagefile)

    # get bounding box of the hole
    validPoints = np.where(mask != 0)
    boundary = np.min(validPoints[0]), np.max(validPoints[0]), np.min(validPoints[1]), np.max(validPoints[1])

    # get search boundary (3 times larger than the hole)
    height = boundary[1] - boundary[0]
    minRow = boundary[0] - height
    if minRow < 0:
        minRow = 0
    maxRow = boundary[1] + height
    if maxRow > image.shape[0] - 1:
        maxRow = image.shape[0] - 1
    width = boundary[3] - boundary[2]
    minCol = boundary[2] - width
    if minCol < 0:
        minCol = 0
    maxCol = boundary[3] + width
    if maxCol > image.shape[1] - 1:
        maxCol = image.shape[1] - 1
    TAU = max(maxRow - minRow, maxCol - minCol) / 15

    # get patches within search boundary
    indices, patches = [], []
    rows, cols, _ = image.shape
    for i in range(minRow, maxRow - PATCH_SIZE, PATCH_SIZE // 2):
        for j in range(minCol, maxCol - PATCH_SIZE, PATCH_SIZE // 2):
            if i not in range(boundary[0] - PATCH_SIZE, boundary[1]) and j not in range(
                    boundary[2] - PATCH_SIZE, boundary[3]):
                indices.append([i + PATCH_SIZE // 2, j + PATCH_SIZE // 2])
                patches.append(image[i:i + PATCH_SIZE, j:j + PATCH_SIZE].flatten())

    # get offsets
    offsets = GetOffsets2(patches, indices)
    # get k dominant offsets by using 2d histogram
    kDominantOffset = GetKDominantOffsets(offsets, 90, maxRow - minRow, maxCol - minCol)
    # optimize the energy function by graph cuts
    sites, optimalLabels = graphcut.graphcut(image2, mask, kDominantOffset, sd)
    # use optimal points to fill the hole
    finalImg = image2
    for i in range(len(sites)):
        j = optimalLabels[i]
        try:
            finalImg[sites[i][0], sites[i][1]] = image2[sites[i][0] + kDominantOffset[j][0], sites[i][1] + kDominantOffset[j][1]]
        except:
            print(sites[i][0] + kDominantOffset[j][0], sites[i][1] + kDominantOffset[j][1])

    cv2.imwrite(imagefile + "_Complete.png", finalImg)


if __name__ == "__main__":
    patch(sys.argv[1], sys.argv[2])
