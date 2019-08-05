import numpy as np

def sliding_window(ing, size, mask):
    windows = []
    h, w, *c = ing.shape
    wsize = size//2
    for i in range(wsize, h-wsize):
        for j in range(wsize, w-wsize):
            if mask[i,j] > 0:
                windows.append(ing[i-wsize:i+wsize+1, j-wsize:j+wsize+1])
    return np.array(windows)

def sliding_window_gt(ing, size, mask):
    windows = []
    h, w, *c = ing.shape
    wsize = size//2
    for i in range(wsize, h-wsize):
        for j in range(wsize, w-wsize):
            if mask[i, j] > 0:
                windows.append(ing[i, j])
    return windows

def categorize(ing, nclasses=2):
    out = np.zeros((*ing.shape, nclasses))
    for i in range(nclasses):
        out[:,i] = (ing[:] == i).astype(int)
    return out

def categorize2d(ing, nclasses=2):
    out = np.zeros((*ing.shape, nclasses))
    for i in range(nclasses):
        out[:,:,i] = (ing[:,:] == i).astype(int)
    return out

def extract_patches(img, size):
    H, W, *c = img.shape
    h = H//size[0]
    w = W//size[1]
    patches = []
    for i in range(h):
        for j in range(w):
            patches.append(img[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]])
    return np.array(patches)

def to_categorical(patches):
    n, H, W = patches.shape
    segmentation = np.zeros((n,H,W,2))
    for i in range(n):
        segmentation[i,:,:,0] = (patches[i,:,:] == 0).astype(int)
        segmentation[i,:,:,1] = (patches[i,:,:] > 0).astype(int)
    return segmentation
