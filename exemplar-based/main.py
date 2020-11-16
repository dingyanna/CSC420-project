from matplotlib import pyplot as plt
import cv2
import numpy as np

def main(f, fill_front, region, k=9):
    """
    Find the most similar pixel in the source region and copy it to the 
    target region iteratively. Return the propogated image.
    
    Parameters:
        f: input image
        fill_front: an array of pixels specifying the contour of the inpainted
                    region
        k: the size of the patches
    """
    # assert k is odd
    assert(k % 2 == 1)
    
    # Initialize confidence
    H = f.shape[0] # height
    W = f.shape[1] # width
    dw = int((k - 1) / 2)
    C = np.ones((H,W))
    C[fill_front[0][0]:fill_front[1][1], fill_front[0][0]:fill_front[1][1]] = 0
    
    # Preprocess image
    f_gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    f_x = cv2.Sobel(f_gray, cv2.CV_64F, 1, 0)
    f_y = cv2.Sobel(f_gray, cv2.CV_64F, 0, 1)
    f_x = f_x / 255.
    f_y = f_y / 255.
    f_grad_mag = np.sqrt(f_x * f_x + f_y * f_y)
    

    fill_front_cp = fill_front.copy()
    while len(fill_front) != 0:
        print(fill_front)
        max_priority = float('-inf')
        patch = None
        # Compute patch_priority = confidence_term + data_term
        for (i, j) in fill_front:
            # Compute confidence term 
            confidence = 0
            for u in range(i-dw, i+dw+1):
                for v in range(j-dw, j+dw+1):
                    if u not in range(H) or v not in range(W):
                        continue 
                    if (u, v) not in region:
                        confidence += 1
            confidence /= (k ** 2)
        
            # Compute data term 
            data = f_grad_mag[i, j]
            
            priority = confidence + data  
            if priority > max_priority:
                patch = (i, j)
                max_priority = priority   
        
        # Propogate 
        min_diff = float('inf')
        source_patch = None
        for i in range(dw, H-dw):
            for j in range(dw, W-dw):
                diff = 0
                for u in range(-dw, dw+1):
                    for v in range(dw, dw+1):
                        if (patch[0]+u) not in range(H) or (patch[1]+v) not in range(W):
                            continue
                        diff += np.sum((f[i+u, j+v] - f[patch[0]+u, patch[1]+v]) ** 2)
                if diff < min_diff:
                    source_patch = (i, j)
                    min_diff = diff
        
        # Copy pixels and update confidence level
        for u in range(-dw, dw+1):
            for v in range(-dw, dw+1):
                if u not in range(H) or v not in range(W):
                    continue 
                if (patch[0]+u, patch[1]+v) in region:
                    f[patch[0]+u, patch[1]+v] = f[source_patch[0]+u, source_patch[1]+v]
                    C[patch[0]+u, patch[1]+v] = C[patch[0], patch[1]]
                    region[(patch[0]+u, patch[1]+v)] = 1
        
        count = sum(region.values())
        print(count)
        
        fill_front.remove(patch)
        # Update fill_front if possible
        if len(fill_front) == 0 and count < len(region):
            fill_front = []
            fill_front_cp = fill_front.copy()
                
    return f



if __name__ == '__main__':
    f = cv2.imread('../images/test.jpg')
    # f_cle = cv2.cvtColor(f, f, cv2.CV_RGB2Lab)
    # print(f_cle.shape)
    region = {}
    for i in range(7, 10):
        for j in range(7, 10):
            region[(i,j)] = 0
    fill_front = [(7, 7), (8,7), (9,7), (9,8), (9, 9), (8, 9), (7, 9)]
    f_inpainted = main(f, fill_front, region, 3)
    plt.imsave('../results/test.png', cv2.cvtColor(f_inpainted, cv2.COLOR_BGR2RGB))
    
