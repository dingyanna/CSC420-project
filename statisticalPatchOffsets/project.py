import cv2
import numpy as np
from sklearn.neighbors import KDTree
import sys
import graphcut
import matplotlib.pyplot as plt

# parameters to set the size of each patch
PATCH_SIZE = 8
# the kernel size of the gaussian blurring.
KERNEL_SIZE = 9


"""
committer:    Peizhi ZHang
   I consulted with https://github.com/Pranshu258/Image_Completion/
   My version is a fresh new version, independent of its code, implemented from scratch 
   
   1. For step1 of the algorithm I used sklearn.neighbors.KDTree instead of the KDTree provided 
   in the repo to get offsets
   my version is faster. In their codes, they used PCA to reduce dimension of patches before patches 
   were sent into KDTree because the KDTree they provided is very slow and may crash.
   
   2. For step2 of the algorithm I used gaussian filter to smooth the histogram. After smoothing, 
   I split the patches into 16 * 16 windows, and get the local maximum, and get the K maximum offsets. 
   Their version dilates the histogram and get the k dominant offsets without splitting into windows. 
   my version is an exact implementation of the paper. 
   
   3. For step3 of the algorithm, I used gco_wrapper library to implement graphcut optimization, 
   where the api calls are: create_general_graph() -> set_site_data_cost() -> set_all_neighbors() -> 
   set_smooth_cost_function() -> swap() -> get_labels(). Their code uses a library called maxflow 
   but the functionalities of maxflow are deficient 
   
   4. For results, their results are far from acceptable, while my version is indeed acceptable. 
   
   5. I have been stammering since childhood and it gets worse when I am nervous,
   so during the oral defence I didn't say too many words.
   
"""


def Matchingsimilarpatches(patches, indices, TAU):
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
    print("gMatchingsimilarpatches done", np.array(offsets).shape)
    return offsets

def Findingdominantoffsets(offsets, K, height, width):
    #Given all the offsets s(x), we compute their statistics by a 2-d histogram h(u; v):
    rows, cols = [], []
    for i in range(len(offsets)):
        rows.append(offsets[i][0])
        cols.append(offsets[i][1])
    bin_rows, bin_cols = [], []
    for i in range(np.min(rows), np.max(rows)):
        bin_rows.append(i)
    for i in range(np.min(cols), np.max(cols)):
        bin_cols.append(i)
    hist, xedges, yedges = np.histogram2d(rows, cols, bins=[bin_rows, bin_cols])

    # The histogram in (2) is further smoothed by a Gaussian filter
    hist = cv2.GaussianBlur(hist, (KERNEL_SIZE, KERNEL_SIZE), np.sqrt(2))

    # A peak in the smoothed histogram is a bin whose magnitude is locally maximal in a d×d window(d=8)
    localhist = np.zeros(hist.shape)
    d = 16
    for r in range(0, hist.shape[0], d):
        for c in range(0, hist.shape[1], d):
            r2, c2 = r + d, c + d
            if r2 > hist.shape[0]:
                r2 = hist.shape[0]
            if c2 > hist.shape[1]:
                c2 = hist.shape[1]
            idx = np.argmax(hist[r:r2, c:c2])
            maxr, maxc = idx // (c2 - c), idx % (c2 - c)
            #print("r,r2,c,c2:", r, r2, c, c2)
            #print("idx,maxr,maxc", idx, maxr, maxc)
            localhist[r + maxr, c + maxc] = hist[maxr + r, maxc + c]#np.sum(hist[r:r2, c:c2])#np.sum(hist[r:r2, c:c2] != 0)
    print(localhist.flatten().shape, hist.shape)

    #We pick out the K highest peaks of this histogram
    threshold = sorted(localhist.flatten(), reverse=True)[K]
    peaks, freq = [], []
    for r in range(0, localhist.shape[0]):
        for c in range(0, localhist.shape[1]):
            if localhist[r][c] >= threshold:
                peaks.append([localhist[r][c], xedges[r], yedges[c]])
                #freq.append(localhist[r][c])
    peaks = np.array(sorted(peaks, reverse=True), dtype="int32")
    freq = peaks[:, 0]
    peaks = peaks[:, 1:3]
    #print(peaks.shape)

    # plot the K highest peaks of this histogram
    f = plt.figure()
    subf = f.add_subplot(111, projection='3d')
    subf.scatter(peaks[:, 1], peaks[:, 0], freq)
    subf.set_xlabel('u')
    subf.set_ylabel('v')
    subf.set_zlabel('frequency')
    subf.set_xlim([-height, height])
    subf.set_ylim([-width, width])
    plt.show()
    return peaks

def patch(imagefile, maskfile):
    # read in image
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(imagefile)

    mask = cv2.imread(maskfile, cv2.IMREAD_GRAYSCALE)

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
    #The threshold τ in Eq.(1) is 1/15 of the max of the rectangle’s width and height.
    TAU = max(maxRow - minRow, maxCol - minCol) / 15

    # get patches within search boundary
    indices, patches = [], []
    rows, cols = image.shape
    for i in range(minRow, maxRow - PATCH_SIZE, PATCH_SIZE // 2):
        for j in range(minCol, maxCol - PATCH_SIZE, PATCH_SIZE // 2):
            if i not in range(boundary[0] - PATCH_SIZE, boundary[1]) or j not in range(
                    boundary[2] - PATCH_SIZE, boundary[3]):
                indices.append([i + PATCH_SIZE // 2, j + PATCH_SIZE // 2])
                patches.append(image2[i:i + PATCH_SIZE, j:j + PATCH_SIZE].flatten())

    print(len(patches))
    # step 1: Matching similar patches
    offsets = Matchingsimilarpatches(np.array(patches), indices, TAU)

    # step 2: Finding dominant offsets.
    kDominantOffset = Findingdominantoffsets(offsets, 90, maxRow - minRow, maxCol - minCol)

    # step 3: Combining Shifted Images via Optimization
    sites, bestLabals = graphcut.graphcut(image2, mask, kDominantOffset)

    # use optimal points to fill the hole
    for i in range(len(sites)):
        j = bestLabals[i]
        try:
            image2[sites[i][0], sites[i][1]] = image2[sites[i][0] + kDominantOffset[j][0], sites[i][1] + kDominantOffset[j][1]]
        except:
            print(sites[i][0] + kDominantOffset[j][0], sites[i][1] + kDominantOffset[j][1])

    cv2.imwrite(imagefile + "-filled.png", image2)


if __name__ == "__main__":
    patch(sys.argv[1], sys.argv[2])
