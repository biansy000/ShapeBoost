import cv2
import numpy as np
import random

from copy import deepcopy
import scipy
from scipy import ndimage
from numba import njit


class TrimapError(Exception):
    """
    Error when creating matting trimap.
    """
    def __init__(self, err):
        super(TrimapError, self).__init__(err)


def normalize(heatmap):
    mm = np.max(heatmap[heatmap>=0])
    heatmap[heatmap<0] = mm
    #heatmap[heatmap<400] = (heatmap[heatmap<400]/20)**2
    #heatmap = -np.log(heatmap/mm)
    heatmap =  (1-heatmap/mm)**3
    mm = np.max(heatmap)
    heatmap = heatmap/mm *255
    return heatmap


def getTrimap(outline):
    # out -> in: 0, 1, 2, 3, 0
    trimap = [] #从左到右是从外到内
    for i in range(1, 4, 1):
        trimap.append(np.stack(np.where(outline==i), axis=1))
    return trimap


def getRings(img, trimap, posflags=None):
    width = img.shape[0]
    height = img.shape[1]

    rings = []
    res_posflags = []
    ratios = []
    for i in range(3):
        x, y = trimap[i][:, 0], trimap[i][:, 1]
        if posflags is not None and len(posflags[i]):
            posflag = posflags[i]
        else:
            posflag = (x < width) * (x >= 0) * (y < height) * (y >= 0)

        if posflag.sum() > 10:
            x = x[posflag]
            y = y[posflag]
            rings.append(img[x, y])
            res_posflags.append(posflag)
    
            ratios.append(posflag.sum() / trimap[i].shape[0])
        else:
            return [], [], []
        
    return rings, res_posflags, ratios


def _get_trimap(single_mask, kernel_size=5):
    # single_mask: [H, W], value [0 - 255]

    mask_list = [single_mask]
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    trimap = np.zeros((single_mask.shape[0], single_mask.shape[1]), dtype=np.int32)

    for i in range(1, 4):
        mask_list.append(cv2.dilate(single_mask.astype(np.float64), kernel, iterations=i * 2))

    for i in range(3, -1, -1):
        trimap[mask_list[i] > 200] = 4 - i # out -> in: 0, 1, 2, 3, 0

    return trimap


def translateTrimap(trimap, oripos, augpos):
    augTrimap = deepcopy(trimap)
    oripos = np.array(oripos)
    for k in range(3):
        augTrimap[k] = trimap[k] - oripos + augpos
    return augTrimap


def Euclidean_Distance(vector1, vector2):
    return np.sum(np.sqrt(np.sum(np.square(vector1-vector2), axis=1)))



def getHeatpoint(image, oriTrimap, oriRings, background, oripos, augpos, config=[0.25, 0.35, 0.4]):
    augTrimap = translateTrimap(oriTrimap, oripos, augpos)
    augRings, augPosflags, augRatios = getRings(background, augTrimap)
    if not augRings:
        return -1

    oriRings, _, _ = getRings(image, oriTrimap, augPosflags)
    heatPoint = 0
    for i in range(3):
        if len(oriRings[i]) and len(augRings[i]):
            matori = oriRings[i].astype(int)
            mataug = augRings[i].astype(int)
            ed = Euclidean_Distance(matori, mataug)/255
            heatPoint += config[i] * ed * (10 - 9 * augRatios[i])
    return heatPoint


def _get_paste_pos(image, human_mask, trimap, center, shrink=20, ratio=0.4):
    orishape = image.shape[:2]
    desshape = (int(image.shape[0] / shrink), int(image.shape[1] / shrink))
    image = cv2.resize(image, (desshape[1], desshape[0]))
    # background = cv2.resize(background, (desshape[1], desshape[0]))
    human_mask = cv2.resize(human_mask, (desshape[1], desshape[0]))
    background = cv2.inpaint(image, (human_mask * 255).astype(np.uint8), 4, cv2.INPAINT_NS)
    trimap = cv2.resize(trimap.astype(np.uint8), (desshape[1], desshape[0]))
    oripos = [int(center[1] / shrink), int(center[0] / shrink)]

    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    oriTrimap = getTrimap(trimap)
    # print('oriTrimap', oriTrimap.shape)
    oriRings = getRings(image, oriTrimap)

    res = []
    for i in range(background.shape[0]):
        for j in range(background.shape[1]):
            heatPoint = getHeatpoint(image, oriTrimap, oriRings, background, oripos, [i, j], config=[0.25, 0.35, 0.4])
            res.append([i,j,heatPoint])
    for point in res:
        heatmap[point[0]][point[1]] = point[2]
    heatmap = normalize(heatmap)
    heatmap = cv2.resize(heatmap, (orishape[1], orishape[0]))

    # pick a position
    poses = np.stack(np.where(heatmap>200), axis=1)
    if len(poses)==0:
        poses = np.stack(np.where(heatmap>150), axis=1)

    if len(poses)==0:
        return None, None

    choice = np.random.choice(range(len(poses)))
    pos = poses[choice]

    return heatmap, pos


def gettrimap_matting(mask, k):
    """
    Compute matting trimap from given mask.
    :param mask: binary ground truth mask
    :param k: number of extended pixels
    :return: matting trimap. 255 for groundtruth foreground, 127 for uncertain area, 0 for ground truth background
    """
    #np.set_printoptions(threshold=np.nan)
    kernel = np.ones((2 * k + 1, 2 * k + 1), dtype=np.int32)
    trimap = ndimage.convolve(mask, kernel, mode='constant')

    trimap = (trimap > 0) * 127
    trimap[mask > 0] = 255

    if trimap.max() != 255 or trimap.min() != 0:
        raise TrimapError('matting trimap failed.')
    return trimap.astype(np.uint8)


def alpha_matting(image, human_mask):

    trimap = gettrimap_matting(human_mask, 5)
    image = image / 255.0
    trimap = trimap / 255.0
    
    # make matting laplacian
    i,j,v = closed_form_laplacian(image)
    h,w = trimap.shape
    L = scipy.sparse.csr_matrix((v, (i, j)), shape=(w*h, w*h))

    # build linear system
    A, b = make_system(L, trimap)

    # solve sparse linear system
    print("solving linear system...")
    alpha = scipy.sparse.linalg.spsolve(A, b).reshape(h, w)

    # stack rgb and alpha
    cutout = np.concatenate([image, alpha[:, :, np.newaxis]], axis=2)

    # clip and convert to uint8 for PIL
    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    alpha  = np.clip(alpha * 255, 0, 255).astype(np.uint8)

    return cutout, alpha


@njit
def closed_form_laplacian(image, epsilon=1e-7, r=1):
    h,w = image.shape[:2]
    window_area = (2*r + 1)**2
    n_vals = (w - 2*r)*(h - 2*r)*window_area**2
    k = 0
    # data for matting laplacian in coordinate form
    i = np.empty(n_vals, dtype=np.int32)
    j = np.empty(n_vals, dtype=np.int32)
    v = np.empty(n_vals, dtype=np.float64)

    # for each pixel of image
    for y in range(r, h - r):
        for x in range(r, w - r):

            # gather neighbors of current pixel in 3x3 window
            n = image[y-r:y+r+1, x-r:x+r+1]
            u = np.zeros(3)
            for p in range(3):
                u[p] = n[:, :, p].mean()
            c = n - u

            # calculate covariance matrix over color channels
            cov = np.zeros((3, 3))
            for p in range(3):
                for q in range(3):
                    cov[p, q] = np.mean(c[:, :, p]*c[:, :, q])

            # calculate inverse covariance of window
            inv_cov = np.linalg.inv(cov + epsilon/window_area * np.eye(3))

            # for each pair ((xi, yi), (xj, yj)) in a 3x3 window
            for dyi in range(2*r + 1):
                for dxi in range(2*r + 1):
                    for dyj in range(2*r + 1):
                        for dxj in range(2*r + 1):
                            i[k] = (x + dxi - r) + (y + dyi - r)*w
                            j[k] = (x + dxj - r) + (y + dyj - r)*w
                            temp = c[dyi, dxi].dot(inv_cov).dot(c[dyj, dxj])
                            v[k] = (1.0 if (i[k] == j[k]) else 0.0) - (1 + temp)/window_area
                            k += 1
        print("generating matting laplacian", y - r + 1, "/", h - 2*r)

    return i, j, v

def make_system(L, trimap, constraint_factor=100.0):
    # split trimap into foreground, background, known and unknown masks
    is_fg = (trimap > 0.9).flatten()
    is_bg = (trimap < 0.1).flatten()
    is_known = is_fg | is_bg
    is_unknown = ~is_known

    # diagonal matrix to constrain known alpha values
    d = is_known.astype(np.float64)
    D = scipy.sparse.diags(d)

    # combine constraints and graph laplacian
    A = constraint_factor*D + L
    # constrained values of known alpha values
    b = constraint_factor*is_fg.astype(np.float64)

    return A, b


def get_heatmap(image, human_mask, bbox):
    # bbox: [x1, y1, x2, y2]
    # background = cv2.inpaint(image, (human_mask * 255).astype(np.uint8), 5, cv2.INPAINT_NS)
    
    desshape = (int(image.shape[0] / 6), int(image.shape[1] / 6))
    human_mask = cv2.resize(human_mask, (desshape[1], desshape[0]))

    trimap = _get_trimap(human_mask * 255)
    # print('trimap', trimap.shape)

    x1, y1, x2, y2 = bbox
    center = [(x1 + x2) / 2, (y1 + y2) / 2]
    heatmap, pos = _get_paste_pos(image, human_mask, trimap, center, 20)

    return heatmap, pos

