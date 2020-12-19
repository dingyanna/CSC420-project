
from matplotlib import pyplot as plt
import cv2
import numpy as np
import time
import sys
import argparse


def get_correct_bounds(lbs, ubs, H, W):
    # lower bounds
    if lbs[2] < 0:
        lbs[0] -= lbs[2]
        lbs[2] = 0 
    if lbs[3] < 0:
        lbs[1] -= lbs[3]
        lbs[3] = 0 
    # upper bounds
    if ubs[2] > H:
        ubs[0] -= (ubs[2] - H)
        ubs[2] = H 
    if ubs[3] > W:
        ubs[1] -= (ubs[3] - W)
        ubs[3] = W
    return lbs, ubs

def detect_fill_front(mask):
    H = mask.shape[0]
    W = mask.shape[1]
    right_shift, down_shift = np.zeros(mask.shape), np.zeros(mask.shape)
    right_shift[:, 1:] = mask[:, :W - 1]
    down_shift[1:, :] = mask[:H - 1, :]
    vertical = np.argwhere((right_shift - mask) != 0)
    horizontal = np.argwhere((down_shift - mask) != 0)

    fill_front = np.unique(np.concatenate((vertical,horizontal)), axis=0)
    return fill_front.tolist()

def update_image_statistics(f, mask):
    f_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    if len(f.shape) == 2:
        f_color = f_gray
    else:
        f_color = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
    f_x = cv2.Sobel(f_gray, cv2.CV_64F, 1, 0)
    f_y = cv2.Sobel(f_gray, cv2.CV_64F, 0, 1)
    mask_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0)
    mask_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1)
    return f_color, f_x, f_y, mask_x, mask_y



def main(f, mask, k=9):
    """
    Find the most similar pixel in the source region and copy it to the 
    target region iteratively. Return the propogated image.
    
    Parameters:
        f: input image
        mask: specifies the region to be inpainted in white with a black 
                background
        k: the size of the patches
    """
    # assert k is odd
    assert(k % 2 == 1)

    H, W = f.shape[0], f.shape[1]
    dw = int((k - 1) / 2)

    # Initialize confidence
    C = np.ones((H, W)) # padded
    C[mask == 255] = 0

    while mask.sum() > 0:
        fill_front = detect_fill_front(mask)

        f_color, f_x, f_y, mask_x, mask_y = update_image_statistics(f, mask)
    
        max_priority = float('-inf')
        patch = None
        max_confidence = None
        # Compute patch_priority = confidence_term + data_term
        for [i, j] in fill_front:
            # Compute confidence term 
            confidence = 0
            for u in range(-dw, dw+1):
                for v in range(-dw, dw+1):
                    if i+u not in range(H) or j+v not in range(W):
                        continue 
                    if mask[i+u, j+v] == 0:
                        confidence += C[i+u, j+v]
            confidence /= (k ** 2)
            
            # Compute data term 
            isophote = [f_y[i, j], - f_x[i, j]]
            normal = np.array([mask_y[i, j], - mask_x[i, j]]) # vector orthogonal to contour
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm

            data = abs(isophote[0] * normal[0] + isophote[1] * normal[1]) / 255.

            priority = confidence * data
           
            if priority > max_priority:
                patch = [i, j]
                max_confidence = confidence
                max_priority = priority   
        C[patch[0], patch[1]] = max_confidence

        # Propogate 
        min_diff = float('inf')
        source = None
        masked = mask[patch[0]-dw:patch[0]+dw+1, patch[1]-dw:patch[1]+dw+1] 
        for i in range(dw, H-dw):
            for j in range(dw, W-dw):
                if mask[i-dw:i+dw+1, j-dw:j+dw+1].sum() > 0:
                    # skip the regions not entirely in source region
                    continue
                
                lbs = [i-dw, j-dw, patch[0]-dw, patch[1]-dw] # lower bounds
                ubs = [i+dw+1, j+dw+1, patch[0]+dw+1, patch[1]+dw+1] # upper bounds
                lbs, ubs = get_correct_bounds(lbs, ubs, H, W)
                
                diff = f_color[lbs[0]:ubs[0], lbs[1]:ubs[1]] - \
                        f_color[lbs[2]:ubs[2], lbs[3]:ubs[3]]

                diff = diff[masked == 0] # only consider difference in non-filled pixels
                diff = np.square(diff).sum()
                if diff < min_diff:
                    source = [i, j]
                    min_diff = diff
    
        for u in range(-dw, dw+1):
            for v in range(-dw, dw+1):
                if patch[0]+u not in range(H) or patch[1]+v not in range(W):
                    continue 
                if mask[patch[0]+u, patch[1]+v] == 255:
                    f[patch[0]+u, patch[1]+v] = f[source[0]+u, source[1]+v]
                    C[patch[0]+u, patch[1]+v] = C[patch[0], patch[1]]
                    mask[patch[0]+u, patch[1]+v] = 0

    return f

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program inpaint '\
        'a given image with a provided mask and patch size.')
    parser.add_argument('input_file_name', 
                        nargs=1, 
                        help='input image\'s file name')
    parser.add_argument('mask', 
                        nargs=1, 
                        help='mask image\'s file name')
    parser.add_argument('output_file_name', 
                        nargs=1, 
                        help='output image\'s file name')
    parser.add_argument('patchsize',
                        nargs=1, 
                        help='patch\'s side length')
    
    args = parser.parse_args()
    input_image = args.input_file_name[0]
    mask = args.mask[0]
    output_image = args.output_file_name[0]
    k = int(args.patchsize[0])
    
    f = cv2.imread(input_image)
    mask = cv2.imread(mask)
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask[mask > 0] = 255.
    
    f_inpainted = main(f, mask, k)
    
    plt.imsave(output_image, cv2.cvtColor(f_inpainted, cv2.COLOR_BGR2RGB))
