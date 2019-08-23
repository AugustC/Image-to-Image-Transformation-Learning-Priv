import numpy as np
import scipy.misc as m
import keras

class DataGen(keras.utils.Sequence):

    def __init__(self, data_lists=None, paths=None, window=-1, batch_size=256, n_outputs=2, shuffle=True, mode='RGB', load_all=True, imsize=[-1,-1], sampling=-1.0):
        # data_lists <- dict. List of names of the images, ground truth and (optional) masks.
        # paths <- dict. Paths to images, ground truth and (optional) masks
        # window <- int. Window size
        # load_all <- bool. True if all of the data can be load into memory
        self.images_list = data_lists['images']
        self.gt_list = data_lists['gt']
        self.mask_list = data_lists.get('mask')
        self.path_X = paths["images"]
        self.path_y = paths["gt"]
        self.path_mask = paths.get("mask")
        self.window_side = int(window/2)
        self.batch_size = batch_size
        self.n_outputs = n_outputs
        self.imsize = imsize
        self.sampling = sampling
        self.mask_id = self.get_mask_id(self.mask_list)
        self.mode = mode
        self.load_all = load_all
        if self.load_all:
            self.data = self.load_data(images_list, gt_list)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.mask_id)/self.batch_size))

    def __getitem__(self, index):
        ind = self.mask_id[index*self.batch_size:(index+1)*self.batch_size]
        if self.load_all:
            X,y = self.get_batch(ind.astype(int))
        else:
            X,y = self.datagen(ind.astype(int))
        return X,y

    def on_epoch_end(self):
        p = np.random.permutation(len(self.mask_id))
        self.mask_id = self.mask_id[p]

    def datagen(self, indices):
        X = []
        y = []
        ind = indices[indices[:,2].argsort()] #sort indices by image index.
        previous_img = -1
        win = self.window_side
        for i,j,id_img in ind:
            if id_img != previous_img:
                previous_img = id_img
                # Special case when only reading the green channel
                if self.mode == 'G':
                    img = m.imread(self.path_X+self.images_list[id_img], mode='RGB')
                    img = img[:,:,1]
                else:
                    img = m.imread(self.path_X+self.images_list[id_img], mode=self.mode)
                gt = m.imread(self.path_y+self.gt_list[id_img], mode='L')

            H, W, *channels = img.shape
            # If i and j are within the image field
            if i - win > 0 and i + win + 1 < H and j - win > 0 and j + win + 1 < W:
                X.append(img[i-win:i+win+1, j-win:j+win+1])
                y.append(gt[i,j])

        X = np.asarray(X)
        if len(X.shape) == 3:
            X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
        if self.n_outputs == 2:
            y = (np.asarray(y) > 0).astype(int)
        y = keras.utils.to_categorical(y, self.n_outputs)
        y = np.reshape(y, (y.shape[0], 1, 1, y.shape[1]))
        return X, y

    def load_data(self, images_list, gt_list):
        imgs = []
        labels = []
        for img_name, gt_name in zip(images_list, gt_list):
            if self.mode == 'G':
                img = m.imread(self.path_X+img_name, mode='RGB')
                img = img[:,:,1]
            else:
                img = m.imread(self.path_X+img_name, mode=self.mode)
            gt = m.imread(self.path_y+gt_name, mode='L')
            imgs.append(img)
            labels.append(gt)
        return imgs, labels

    def get_batch(self, ind):
        X = []
        y = []
        win = self.window_side
        for i, j, id_img in ind:
            H, W, *channels = self.data[0][id_img].shape
            if i - win > 0 and i + win + 1 < H and j - win > 0 and j + win + 1 < W:
                X.append(self.data[0][id_img][i-win:i+win+1, j-win:j+win+1])
                y.append(self.data[1][id_img][i,j])
        X = np.asarray(X)
        if len(X.shape) == 3:
            X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
        if self.n_outputs == 2:
            y = (np.asarray(y) > 0).astype(int)
        y = keras.utils.to_categorical(y, self.n_outputs)
        y = np.reshape(y, (y.shape[0], 1, 1, y.shape[1]))
        return X, y

    def get_mask_id(self, mask_list):
        # Masks can be used to define pixels to be extracted
        win = self.window_side
        mask_id = np.array([]).reshape(0,3)
        n = len(self.images_list)

        # If there is a mask for each image
        if mask_list:
            for i, mask in enumerate(mask_list):
                image = m.imread(self.path_mask+mask, mode='L')
                idx, idy = np.nonzero(image[win:-win, win:-win])
                maskid = np.ones(len(idx), dtype=int)*i
                mask_id = np.concatenate((mask_id, np.dstack((idx,idy,maskid))[0]))
        else:
            # If images are the same size, we can easily create mask_ids
            if self.imsize[0] > 0:
                H, W = self.imsize
                mask_id = np.array(np.meshgrid(np.arange(H), np.arange(W), np.arange(n))).T.reshape(-1,3)

            # Otherwise, we need to read every image and find their size
            else:
                for i, img in enumerate(self.images_list):
                    image = m.imread(self.path_X+img, mode='L')
                    H, W, *C = image.shape
                    ids = [[h,w,i] for h in range(H) for w in range(W)]
                    mask_id = np.concatenate((mask_id, np.array(ids)))

        if sampling > 0:
            n_ids = np.floor(len(mask_id) * self.sampling)
            mask_id = mask_id[:n_ids]
        return mask_id

class DataGenPatches(DataGen):

    def __init__(self, l_patch, s_patch, **kw):
        self.l_patch = l_patch
        self.s_patch = s_patch
        DataGen.__init__(self, **kw)

    def get_mask_id(self, mask_list):
        l_patch = self.l_patch
        s_patch = self.s_patch
        mask_id = np.array([]).reshape(0,3)
        n = len(self.images_list)

        if self.imsize[0] > 0:
            H, W = self.imsize
            px = np.arange(l_patch//2+1, H - l_patch//2, s_patch)
            py = np.arange(l_patch//2+1, W - l_patch//2, s_patch)
            pz = np.arange(n)
            mask_id = np.array(np.meshgrid(px, py, pz)).T.reshape(-1,3)
        else:
            for i, img in enumerate(self.images_list):
                image = m.imread(self.path_X+img, mode='L')
                H, W, *C = image.shape
                px = np.arange(l_patch//2+1, H - l_patch//2, s_patch)
                py = np.arange(l_patch//2+1, W - l_patch//2, s_patch)
                ids = np.array(np.meshgrid(px, py, i)).T.reshape(-1,3)
                mask_id = np.concatenate((mask_id, np.array(ids)))

        if self.sampling > 0:
            n_ids = int(len(mask_id) * self.sampling)
            mask_id = mask_id[:n_ids]

        return mask_id

    def datagen(self, indices):
        X = []
        y = []
        ind = indices[indices[:,2].argsort()] #sort indices by image index.
        hl_patch = self.l_patch//2
        hs_patch = self.s_patch//2
        previous_img = -1
        for i,j,id_img in ind:
            if id_img != previous_img:
                previous_img = id_img
                # Special case when only reading the green channel
                if self.mode == 'G':
                    img = m.imread(self.path_X+self.images_list[id_img], mode='RGB')
                    img = img[:,:,1]
                else:
                    img = m.imread(self.path_X+self.images_list[id_img], mode=self.mode)
                gt = m.imread(self.path_y+self.gt_list[id_img], mode='L')

            H, W, *channels = img.shape
            # If i and j are within the image field
            if i - hl_patch > 0 and i + hl_patch + 1 < H and j - hl_patch > 0 and j + hl_patch + 1 < W:
                X.append(img[i-hl_patch:i+hl_patch+1, j-hl_patch:j+hl_patch+1])
                y.append(gt[i-hs_patch:i+hs_patch+1, j-hs_patch:j+hs_patch+1])

        X = np.asarray(X)
        X = X.astype('float32')/255.0
        if len(X.shape) == 3:
            X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
        if self.n_outputs == 2:
            y = (np.asarray(y) > 0).astype(int)
        y = keras.utils.to_categorical(y, self.n_outputs)
#        y = np.reshape(y, (y.shape[0], s_patch, s_patch, y.shape[1]))
        return X, y

