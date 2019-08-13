import numpy as np

def calc_rates(pred, gt):
    gtp = (gt >= 1).astype(int)
    pp = (pred >= 1).astype(int)
    gtn = (gt == 0).astype(int)
    pn = (pred == 0).astype(int)

    TP = (gtp*pp).sum()
    TN = (gtn*pn).sum()
    FP = (gtn*pp).sum()
    FN = (gtp*pn).sum()
    return TP, TN, FP, FN

def precision(pred, gt):
    TP, TN, FP, FN = calc_rates(pred, gt)
    return TP/(TP+FP)

def recall(pred, gt):
    TP, TN, FP, FN = calc_rates(pred, gt)
    return TP/(TP+FN)

def f1(prediction, ground_truth):
    P = precision(prediction, ground_truth)
    R = recall(prediction, ground_truth)
    F = (2*P*R)/(P+R)
    return F

def acc(pred, gt):
    TP, TN, FP, FN = calc_rates(pred, gt)
    return (TP + TN)/(TP + FP + FN + TN)
