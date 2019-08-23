import numpy as np
from skimage.morphology import thin

def calc_rates(pred, gt):
    if type(pred) == list:
        gtp = [(x >= 1).astype(int) for x in gt]
        pp = [(x >= 1).astype(int) for x in pred]
        gtn = [(x < 1).astype(int) for x in gt]
        pn = [(x < 1).astype(int) for x in pred]

    else:
        gtp = (gt >= 1).astype(int)
        pp = (pred >= 1).astype(int)
        gtn = (gt == 0).astype(int)
        pn = (pred == 0).astype(int)

    TP = np.asarray([(gtp[i]*pp[i]).sum() for i in range(len(gtp))])
    TN = np.asarray([(gtn[i]*pn[i]).sum() for i in range(len(gtp))])
    FP = np.asarray([(gtn[i]*pp[i]).sum() for i in range(len(gtp))])
    FN = np.asarray([(gtp[i]*pn[i]).sum() for i in range(len(gtp))])

    return TP, TN, FP, FN

def precision(pred, gt, invert=False):
    TP, TN, FP, FN = calc_rates(pred, gt)
    if invert:
        return (TN/(TN+FN)).mean()
    return((TP/(TP+FP)).mean())

def recall(pred, gt, invert=False):
    TP, TN, FP, FN = calc_rates(pred, gt)
    if invert:
        return (TN/(TN+FP)).mean()
    return (TP/(TP+FN)).mean()

def f1(prediction, ground_truth, invert=False):
    P = precision(prediction, ground_truth, invert=invert)
    R = recall(prediction, ground_truth, invert=invert)
    F = (2*P*R)/(P+R)
    return F

def acc(pred, gt):
    TP, TN, FP, FN = calc_rates(pred, gt)
    return ((TP + TN)/(TP + FP + FN + TN)).mean()

def psnr(pred, gt, invert=True):
    avg_psnr = 0
    C = 1
    if invert:
        TN, TP, FN, FP = calc_rates(pred, gt)
    else:
        TN, TP, FN, FP = calc_rates(pred, gt)
    for i, ing in enumerate(pred):
        H, W, *c = ing[0].shape
        MSE = FP[i]+FN[i]
        MSE = MSE/(H*W)
        PSNR = 10*np.log10(C**2/MSE)
        avg_psnr += PSNR
    return avg_psnr/len(pred)

def pseudoFM(pred, gt, invert=False):
    skels = []
    for i,ing in enumerate(gt):
        sk = thin(ing)
        skels.append(sk)
    if invert:
        pTN, pTP, pFN, pFP = calc_rates(pred, skels)
    else:
        pTP, pTN, pFP, pFN = calc_rates(pred, skels)
    #pseudo_p = pTP / (pFP + pTP)
    pseudo_p = precision(pred, gt, invert=invert)
    pseudo_r = pTP / (pFN + pTP)
    pseudo_F = 2*(pseudo_p * pseudo_r)/(pseudo_p + pseudo_r)
    return pseudo_F.mean()

def drd():
    pass
