"""
Statistics of Patch Offsets for Image Completion - Kaiming He and Jian Sun
A Python Implementation - Pranshu Gupta and Shrija Mishra
"""

import cv2
import sys
import plot
import kdtree
import energy
import operator
import numpy as np
import config as cfg
from time import time
from scipy import ndimage
from sklearn.decomposition import PCA
import graphcut
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from pyflann import *
import scipy


def GetBoundingBox(mask):
    """
    Get Bounding Box for a Binary Mask
    Arguments: mask - a binary mask
    Returns: col_min, col_max, row_min, row_max
    """
    start = time()
    a = np.where(mask != 0)
    bbox = np.min(a[1]), np.max(a[1]), np.min(a[0]), np.max(a[0])
    if cfg.PRINT_BB_IMAGE:
        cv2.rectangle(mask, (bbox[2], bbox[0]), (bbox[3], bbox[1]), (255,255,255), 1)
        cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + cfg.BB_IMAGE_SUFFIX, mask)
    end = time()
    print("GetBoundingBox execution time: ", end - start)
    return bbox

def GetSearchDomain(shape, bbox):
    """
    get a rectangle that is 3 times larger (in length) than the bounding box of the hole
    this is the region which will be used for the extracting the patches
    """
    start = time()
    col_min, col_max = max(0, 2*bbox[0] - bbox[1]), min(2*bbox[1] - bbox[0], shape[1]-1)
    row_min, row_max = max(0, 2*bbox[2] - bbox[3]), min(2*bbox[3] - bbox[2], shape[0]-1)
    end = time()
    print("GetSearchDomain execution time: ", end - start)
    return col_min, col_max, row_min, row_max

def GetPatches(image, bbox, hole):
    """
    get the patches from the search region in the input image
    """
    indices, patches = [], []
    rows, cols, _ = image.shape
    for i in range(int(bbox[2]+cfg.PATCH_SIZE/2), int(bbox[3]-cfg.PATCH_SIZE/2), 4):
        for j in range(int(bbox[0]+cfg.PATCH_SIZE/2), int(bbox[1]-cfg.PATCH_SIZE/2), 4):
            if i not in range(int(hole[2]-cfg.PATCH_SIZE/2), int(hole[3]+cfg.PATCH_SIZE/2)) or j not in range(int(hole[0]-cfg.PATCH_SIZE/2), int(hole[1]+cfg.PATCH_SIZE/2)):
                indices.append([i, j])
                patches.append(image[int(i-cfg.PATCH_SIZE/2):int(i+cfg.PATCH_SIZE/2), int(j-cfg.PATCH_SIZE/2):int(j+cfg.PATCH_SIZE/2)].flatten())

    print("GetPatches : ", len(patches))
    return np.array(indices), np.array(patches, dtype='int64')

def ReduceDimension(patches):
    start = time()
    pca = PCA(n_components=24)
    reducedPatches = pca.fit_transform(patches)
    end = time()
    print("ReduceDimension execution time: ", end - start)
    return reducedPatches

def GetOffsets(patches, indices):
    start = time()
    kd = kdtree.KDTree(patches, leafsize=cfg.KDT_LEAF_SIZE, tau=cfg.TAU)
    dist, offsets = kdtree.get_annf_offsets(patches, indices, kd.tree, cfg.TAU)
    end = time()
    print("GetOffsets execution time: ", end - start)
    return offsets

def GetOffsets2(patches, indices):
    print("build kdtree")
    kd = KDTree(patches, leaf_size=cfg.KDT_LEAF_SIZE)
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
            if offset[0]**2 + offset[1]**2 >= cfg.TAU**2:
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
    hist = cv2.GaussianBlur(hist, (3, 3), np.sqrt(2))
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

def GetOptimizedLabels(image, mask, labels):
    start = time()
    optimizer = energy.Optimizer(image, mask, labels)
    sites, optimalLabels = optimizer.InitializeLabelling()
    #optimalLabels = optimizer.OptimizeLabellingAE(optimalLabels)
    optimalLabels = optimizer.OptimizeLabellingABS(optimalLabels)
    end = time()
    print("GetOptimizedLabels execution time: ", end - start)
    return sites, optimalLabels 

def CompleteImage(image, sites, mask, offsets, optimalLabels):
    failedPoints = mask
    completedPoints = np.zeros(image.shape)
    finalImg = image
    for i in range(len(sites)):
        if mask[sites[i][0], sites[i][1]] == 0:
            continue
        j = optimalLabels[i]
        try:
            finalImg[sites[i][0], sites[i][1]] = image[sites[i][0] + offsets[j][0], sites[i][1] + offsets[j][1]]
            completedPoints[sites[i][0], sites[i][1]] = finalImg[sites[i][0], sites[i][1]]
            failedPoints[sites[i][0], sites[i][1]] = 0
        except:
            print(sites[i][0] + offsets[j][0], sites[i][1] + offsets[j][1])
    return finalImg, failedPoints, completedPoints

def PoissonBlending(image, mask, center):
    src = cv2.imread(cfg.OUT_FOLDER + cfg.IMAGE + "_CompletedPoints.png")
    dst = cv2.imread(cfg.OUT_FOLDER + cfg.IMAGE + "_Complete.png")
    blendedImage = cv2.seamlessClone(src, dst, mask, center, cv2.MIXED_CLONE)
    return blendedImage


def main(imageFile, maskFile):
    """
    Image Completion Pipeline
        1. Patch Extraction
        2. Patch Offsets
        3. Image Stacking
        4. Blending
    """

    sys.setrecursionlimit(10000)
    image = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
    print(image.shape)
    imageR = cv2.imread(imageFile)
    mask = cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)
    bb = GetBoundingBox(mask)
    bbwidth = bb[1] - bb[0]
    bbheight = bb[3] - bb[2]
    sd = GetSearchDomain(image.shape, bb)
    if cfg.TAU == 0:
        cfg.TAU = max(sd[1]-sd[0], sd[3]-sd[2])/15#max(bbwidth, bbheight)/15
        print("TAU: ", cfg.TAU)
    cfg.DEFLAT_FACTOR = image.shape[1]
    indices, patches = GetPatches(imageR, sd, bb)
    reducedPatches = patches#ReduceDimension(patches)
    offsets = GetOffsets2(reducedPatches, indices)
    kDominantOffset = GetKDominantOffsets(offsets, 90, sd[3] - sd[2], sd[1] - sd[0])#image.shape[0], image.shape[1])
    #bbmask = np.zeros(mask.shape)
    #bbmask[bb[2]:bb[3], bb[0]:bb[1]] = 1
    bbmask = cv2.dilate(mask, np.ones((9, 9)))
    sites, optimalLabels = graphcut.graphcut(imageR, mask, kDominantOffset, sd)#GetOptimizedLabels(imageR, mask, kDominantOffset)
    completedImage, failedPoints, completedPoints = CompleteImage(imageR, sites, mask, kDominantOffset, optimalLabels)
    cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "_Complete.png", completedImage)
    cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "_CompletedPoints.png", completedPoints)
    #center = (int(bb[2]+bbwidth/2), int(bb[0]+bbheight/2))
    #blendedImage = PoissonBlending(imageR, mask, center)
    #cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "_blendedImage.png", blendedImage)
    if (np.sum(failedPoints)):
        cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "_Failed.png", failedPoints)
        #main(cfg.OUT_FOLDER + cfg.IMAGE + "_Complete.png", cfg.OUT_FOLDER + cfg.IMAGE + "_Failed.png")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py image_name mask_file_name")
        exit()
    cfg.IMAGE = sys.argv[1].split('.')[0]
    imageFile = cfg.SRC_FOLDER + sys.argv[1]
    print(imageFile)
    maskFile = cfg.SRC_FOLDER + sys.argv[2]
    main(imageFile, maskFile)
    