import matplotlib.pyplot as plt
import scipy.misc as m
import numpy as np
import os
from scipy import ndimage

def load_images(data_lists, paths, window=-1, mode='RGB'):
    images_names = data_lists['images']
    ground_truths = data_lists['gt']
    imgs = []
    gts = []

    for i, ing_nm in enumerate(images_names):
        ing = m.imread(paths['images']+ing_nm, mode=mode)
        gt = m.imread(paths['gt']+ground_truths[i], mode='L')
        if window > 0:
            H, W, *c = ing.shape
            pad_ing = np.zeros((H+window-1, W+window-1, *c), dtype=ing.dtype)
            side = window//2
            pad_ing[side:-side, side:-side] = ing
            pad_ing[:side,side:-side] = ing[:side,:]
            pad_ing[-side:,side:-side] = ing[-side:,:]
            pad_ing[side:-side,:side] = ing[:,:side]
            pad_ing[side:-side,-side:] = ing[:,-side:]
            pad_ing = pad_ing.astype('float32')/255.0
            imgs.append(pad_ing)
        else:
            imgs.append(ing)
        gts.append(gt)

    return imgs, gts

def plot_predicted_images(model, data_lists, paths, window=-1, mode='RGB'):
    imgs, gts = load_images(data_lists, paths, window, mode=mode)

    preds = model.predict(np.asarray(imgs))
    preds = np.argmax(preds, axis=3)

    n = len(preds)
    fig = plt.figure(figsize=(20,30))
    for i, pred in enumerate(preds):
        ax = fig.add_subplot(n,3,i*3+1)
        if mode=='L':
            ax.imshow(imgs[i], cmap='gray')
        else:
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

def postprocess(img):
    med1 = ndimage.median_filter(img, 5)
    med2 = ndimage.median_filter(med1, 3)
    return med2

def post_processing_predicted_image(img):
    process_ing = postprocess(img) 
    return process_ing

def plot_predicted_images_list(model, data_lists, paths, window=-1, mode='RGB'):
    imgs, gts = load_images(data_lists, paths, window, mode=mode)

    preds = []
    gndts = []
    for i, ing in enumerate(imgs):
        if len(ing.shape) < 3:
            reshaped_ing = ing.reshape((1,ing.shape[0], ing.shape[1], 1))
        else:
            reshaped_ing = ing.reshape((1,ing.shape[0], ing.shape[1], ing.shape[2]))
        pred = model.predict(np.asarray(reshaped_ing))
        pred = np.argmax(pred, axis=3)
        processed_pred = post_processing_predicted_image(pred)

        fig = plt.figure(figsize=(20,30))
        ax = fig.add_subplot(1,3,1)
        if mode == 'L':
            ax.imshow(ing, cmap='gray')
        else:
            ax.imshow(ing)
        ax.set_title("Original Image")

        ax = fig.add_subplot(1,3,2)
        ax.imshow(processed_pred[0], cmap='gray')
        ax.set_title('Predicted Post-Processed Image')

        ax = fig.add_subplot(1,3,3)
        ax.imshow(gts[i], cmap='gray')
        ax.set_title('Ground Truth')

        plt.show()

        preds.append(processed_pred)
        gndts.append(gts[i])

    return preds, gndts

def plot_comparison(modelA, modelB, data_lists, paths, window=-1, mode='RGB'):
    imgs, gts = load_images(data_lists, paths, window, mode=mode)

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

def plot_comparison_lists(modelA, modelB, data_lists, paths, window=-1, mode='RGB'):
    imgs, gts = load_images(data_lists, paths, window, mode=mode)

    predictionsA, predictionsB = [],[]
    for i, ing in enumerate(imgs):
        ing = [ing]
        predA = modelA.predict(np.asarray(ing))
        predA = np.argmax(predA, axis=3)
        predictionsA.append(predA)

        predB = modelB.predict(np.asarray(ing))
        predB = np.argmax(predB, axis=3)
        predictionsB.append(predB)

        fig = plt.figure(figsize=(20,30))
        ax = fig.add_subplot(1,3,1)
        ax.imshow(gts[i], cmap='gray')
        ax.set_title("Original Ground Truth")

        ax = fig.add_subplot(1,3,2)
        ax.imshow(predA[0], cmap='gray')
        ax.set_title('Prediction from '+modelA.name)

        ax = fig.add_subplot(1,3,3)
        ax.imshow(predB[0], cmap='gray')
        ax.set_title('Prediction from '+modelB.name)
        plt.show()

    return predictionsA, predictionsB

def preprocess_imgs(func, fn_imgs, dir_imgs, output_dir='processed_imgs/', verbose=0, cmap=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_names = []
    for i, fn_ing in enumerate(fn_imgs):
        ing = m.imread(dir_imgs + fn_ing)
        processed_img = func(ing)
        if verbose:
            fig = plt.figure(figsize=(20,30))
            ax = fig.add_subplot(1,2,1)
            ax.imshow(ing)
            ax.set_title('Original Image')

            ax = fig.add_subplot(1,2,2)
            ax.imshow(processed_img, cmap=cmap)
            ax.set_title('Processed Image')
            plt.show()

        fname = os.path.splitext(fn_ing)[0] + '.png'
        dir_ing = output_dir + os.path.dirname(fname)
        if not os.path.exists(dir_ing) and dir_ing:
            os.makedirs(dir_ing)
        img_names.append(fname)
        m.imsave(output_dir + fname, processed_img)

    return img_names, output_dir
