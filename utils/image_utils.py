import matplotlib.pyplot as plt
import scipy.misc as m
import numpy as np

def load_images(data_lists, paths, window=-1):
    images_names = data_lists['images']
    ground_truths = data_lists['gt']
    imgs = []
    gts = []

    for i, ing_nm in enumerate(images_names):
        ing = m.imread(paths['images']+ing_nm, mode='RGB')
        gt = m.imread(paths['gt']+ground_truths[i], mode='L')
        if window > 0:
            H, W, *c = ing.shape
            pad_ing = np.zeros((H+window-1, W+window-1, *c), dtype=ing.dtype)
            side = window//2
            pad_ing[side:-side, side:-side] = ing
            imgs.append(pad_ing)
        else:
            imgs.append(ing)
        gts.append(gt)

    return imgs, gts

def plot_predicted_images(model, data_lists, paths, window=-1):
    imgs, gts = load_images(data_lists, paths, window)

    preds = model.predict(np.asarray(imgs))
    preds = np.argmax(preds, axis=3)

    n = len(preds)
    fig = plt.figure(figsize=(20,30))
    for i, pred in enumerate(preds):
        ax = fig.add_subplot(n,3,i*3+1)
        ax.imshow(imgs[i])
        ax.set_title("Original Image")

        ax = fig.add_subplot(n,3,i*3+2)
        ax.imshow(pred, cmap='gray')
        ax.set_title('Predicted')

        ax = fig.add_subplot(n,3,i*3+3)
        ax.imshow(gts[i], cmap='gray')
        ax.set_title('Ground Truth')
    plt.show()

    return np.asarray(preds), np.asarray(gts)

def plot_comparison(modelA, modelB, data_lists, paths, window=-1):
    imgs, gts = load_images(data_lists, paths, window)

    predA = modelA.predict(np.asarray(imgs))
    predA = np.argmax(predA, axis=3)

    predB = modelB.predict(np.asarray(imgs))
    predB = np.argmax(predB, axis=3)

    n = len(predA)
    fig = plt.figure(figsize=(20,30))
    for i in range(n):
        ax = fig.add_subplot(n,3,i*3+1)
        ax.imshow(gts[i], cmap='gray')
        ax.set_title("Original Ground Truth")

        ax = fig.add_subplot(n,3,i*3+2)
        ax.imshow(predA[i], cmap='gray')
        ax.set_title('Prediction from '+modelA.name)

        ax = fig.add_subplot(n,3,i*3+3)
        ax.imshow(predB[i], cmap='gray')
        ax.set_title('Prediction from '+modelB.name)
    plt.show()

    return np.asarray(predA), np.asarray(predB)


