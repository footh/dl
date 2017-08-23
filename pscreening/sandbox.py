import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy import misc

from skimage import feature
from docutils.nodes import image

def canny_edge(file='zones/0.png', sigma=1):
    img = misc.imread(file)
    edges = feature.canny(img, sigma=sigma)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), sharex=True, sharey=True)
     
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('noisy image', fontsize=20)
        
    ax2.imshow(edges, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title(f'Canny filter, $\sigma={sigma}$', fontsize=20)
    
    fig.tight_layout()
    
    plt.show()
    return edges

def threshold(file='zones/0.png', t='otsu', img=None):
    from skimage.filters import try_all_threshold
    from skimage.filters import threshold_mean, threshold_li, threshold_otsu

    if img is None:
        img = misc.imread(file)

    fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
    plt.show()
    
    thresh = None
    if t == 'otsu':
        thresh = threshold_otsu(img)
    elif t == 'mean':
        thresh = threshold_mean(img)
    else:
        thresh = threshold_li(img)
        
    binary = img > thresh

    fig, axes = plt.subplots(ncols=2, figsize=(8, 3))
    ax = axes.ravel()

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Original image')
    
    ax[1].imshow(binary, cmap=plt.cm.gray)
    ax[1].set_title('Result')
    
    for a in ax:
        a.axis('off')
    
    plt.show()
    return binary

def skeleton(file='zones/0.png'):
    from skimage.morphology import skeletonize
    from skimage import data

    img = threshold(file)
    
    skeleton = skeletonize(img)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                             sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    
    ax = axes.ravel()
    
    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('original', fontsize=20)
    
    ax[1].imshow(skeleton, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('skeleton', fontsize=20)
    
    fig.tight_layout()
    plt.show()
    
def convex_hull(file='zones/0.png'):
    from skimage.morphology import convex_hull_image
    from skimage import data, img_as_float

    img = threshold(file)
        
    chull = convex_hull_image(img)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.ravel()
    
    ax[0].set_title('Original picture')
    ax[0].imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_axis_off()
    
    ax[1].set_title('Transformed picture')
    ax[1].imshow(chull, cmap=plt.cm.gray, interpolation='nearest')
    ax[1].set_axis_off()
    
    plt.tight_layout()
    plt.show()
    
def edge_ops(file='zones/0.png', use_t=False):
    from skimage.filters import roberts, sobel, scharr, prewitt
    
    img = None
    if use_t:
        img = threshold(file)
    else:
        img = misc.imread(file)

    edge_roberts = roberts(img)
    edge_sobel = sobel(img)

    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))
    
    ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
    ax[0].set_title('Roberts Edge Detection')
    
    ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
    ax[1].set_title('Sobel Edge Detection')
    
    for a in ax:
        a.axis('off')
    
    plt.tight_layout()
    plt.show()
    
def t2():
    """
        uses local/global thresholding, local doesn't seem to work well because the image is already
        pretty binary
    """
    from skimage.morphology import disk
    from skimage.filters import threshold_otsu, rank
    from skimage.util import img_as_ubyte
    
    img = misc.imread('zones/0.png')
    
    radius = 15
    selem = disk(radius)
    
    local_otsu = rank.otsu(img, selem)
    threshold_global_otsu = threshold_otsu(img)
    global_otsu = img >= threshold_global_otsu
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()
    plt.tight_layout()
    
    fig.colorbar(ax[0].imshow(img, cmap=plt.cm.gray),
                 ax=ax[0], orientation='horizontal')
    ax[0].set_title('Original')
    ax[0].axis('off')
    
    fig.colorbar(ax[1].imshow(local_otsu, cmap=plt.cm.gray),
                 ax=ax[1], orientation='horizontal')
    ax[1].set_title('Local Otsu (radius=%d)' % radius)
    ax[1].axis('off')
    
    ax[2].imshow(img >= local_otsu, cmap=plt.cm.gray)
    ax[2].set_title('Original >= Local Otsu' % threshold_global_otsu)
    ax[2].axis('off')
    
    ax[3].imshow(global_otsu, cmap=plt.cm.gray)
    ax[3].set_title('Global Otsu (threshold = %d)' % threshold_global_otsu)
    ax[3].axis('off')
    
    plt.show()
    
def edge2(file='zones/0.png'):
    from skimage.filters import sobel
    from skimage import morphology
    
    img = misc.imread(file)
    
    elevation_map = sobel(img)
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(elevation_map, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title('elevation map')
    ax.axis('off')
    ax.set_adjustable('box-forced')
    plt.show()
    
    markers = np.zeros_like(img)
    markers[img < 30] = 1
    markers[img > 150] = 2
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
    ax.set_title('markers')
    ax.axis('off')
    ax.set_adjustable('box-forced') 
    plt.show()   
    
    segmentation = morphology.watershed(elevation_map, markers)
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title('segmentation')
    ax.axis('off')
    ax.set_adjustable('box-forced')
    plt.show()
    
def dilation(file='zones/0.png', use_t=False, t='otsu', rad=6):
    from skimage.morphology import dilation
    from skimage.morphology import disk
    
    if use_t:
        img = threshold(file, t=t)
    else:
        img = misc.imread(file)

    selem = disk(rad)    
    dilated = dilation(img, selem)
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax1.set_adjustable('box-forced')
    ax2.imshow(dilated, cmap=plt.cm.gray)
    ax2.set_title('dilated')
    ax2.axis('off')
    ax2.set_adjustable('box-forced')
    plt.show()
    
def closing(file='zones/0.png', use_t=False, t='otsu', rad=3):
    from skimage.morphology import closing
    from skimage.morphology import disk
    
    if use_t:
        img = threshold(file, t=t)
    else:
        img = misc.imread(file)

    selem = disk(rad)    
    closed = closing(img, selem)
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax1.set_adjustable('box-forced')
    ax2.imshow(closed, cmap=plt.cm.gray)
    ax2.set_title('closed')
    ax2.axis('off')
    ax2.set_adjustable('box-forced')
    plt.show()

def white_hatting(file='zones/0.png', use_t=False, t='otsu', rad=3):
    from skimage.morphology import white_tophat
    from skimage.morphology import disk
    
    if use_t:
        img = threshold(file, t=t)
    else:
        img = misc.imread(file)

    selem = disk(rad)    
    wh = white_tophat(img, selem)
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax1.set_adjustable('box-forced')
    ax2.imshow(wh, cmap=plt.cm.gray)
    ax2.set_title('white')
    ax2.axis('off')
    ax2.set_adjustable('box-forced')
    plt.show()

def gaussianize(file='zones/0.png', sigma=3, img=None):
    from skimage.filters import gaussian
    
    if img is None:
        img = misc.imread(file)
        
    gauss = gaussian(img, sigma=sigma)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax1.set_adjustable('box-forced')
    ax2.imshow(gauss, cmap=plt.cm.gray)
    ax2.set_title('gaussian')
    ax2.axis('off')
    ax2.set_adjustable('box-forced')
    plt.show()
    
    return gauss

def fill_holes(img):
    from skimage.morphology import reconstruction
    seed = np.copy(img)
    seed[1:-1, 1:-10] = img.max()
    mask = img
    filled = reconstruction(seed, mask, method='erosion')
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax1.set_adjustable('box-forced')
    ax2.imshow(filled, cmap=plt.cm.gray)
    ax2.set_title('filled holes')
    ax2.axis('off')
    ax2.set_adjustable('box-forced')
    plt.show()
    
    return filled
