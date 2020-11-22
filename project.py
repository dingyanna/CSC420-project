import cv2
import numpy as np
from sklearn.decomposition import PCA

PATCH_SIZE = 8
KDT_LEAF_SIZE = 8
TAU = 0

def getsearchboundary(shape, bbox):
    width = bbox[1] - bbox[0]
    col_min, col_max = max(0,   bbox[0] - width), min(bbox[1] + width, shape[1]-1)
    height = bbox[3] - bbox[2]
    row_min, row_max = max(0,  bbox[2] - height), min(bbox[3] + height, shape[0] - 1)
    return col_min, col_max, row_min, row_max

def getpatches(image, bbox, hole):
    indices, patches = [], []
    rows, cols, _ = image.shape
    for i in range(bbox[2] + PATCH_SIZE / 2, bbox[3] - PATCH_SIZE / 2):
        for j in range(bbox[0] + PATCH_SIZE / 2, bbox[1] - PATCH_SIZE / 2):
            if i not in range(hole[2] - PATCH_SIZE / 2, hole[3] + PATCH_SIZE / 2) and j not in range(
                    hole[0] - PATCH_SIZE / 2, hole[1] + PATCH_SIZE / 2):
                indices.append([i, j])
                patches.append(image[i - PATCH_SIZE / 2:i + PATCH_SIZE / 2,
                               j - PATCH_SIZE / 2:j + PATCH_SIZE / 2].flatten())
    return indices, patches

def GetKDominantOffsets(offsets, K, height, width):


def GetOptimizedLabels(image, mask, labels):


# I consulted with https://github.com/Pranshu258/Image_Completion/
def main(imagefile, mask, shape, hole):
    # read in image
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    imageR = cv2.imread()
    # get bounding box of the hole
    validPoints = np.where(mask != 0)
    bbox = np.min(validPoints[0]), np.max(validPoints[0]), np.min(validPoints[1]), np.max(validPoints[1])
    # get search boundary
    sb = getsearchboundary(image.shape, bbox)
    # get patches within search boundary
    indices, patches = getpatches(imageR, sb, bbox)
    # reduce dimension because the kd-tree is less effective for high dimensional data
    pca = PCA(n_components=24)
    reducedPatches = pca.fit_transform(patches)
    # get offsets
    kd = kdtree.KDTree(reducedPatches, leafsize=KDT_LEAF_SIZE, tau=TAU)
    dist, offsets = kdtree.get_annf_offsets(patches, indices, kd.tree, cfg.TAU)
    # get k dominant offsets by using 2d histogram

    # optimize the energy function by graph cuts

    # use optimal points to fill the hole
