from pathlib import Path

import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

def imshow_img_pair(img, mask, t=None):
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(mask)
    if t is not None:
        plt.suptitle(t)
    plt.show()


class ImgOps:

    @classmethod
    def resize(cls, img, dst_shape):
        """
        resizes img while conserving aspect ratio,
        zero pads the short side to make a square,
        then resizes the squared image
        """
        img_shape = img.shape
        is_rgb = len(img_shape) == 3
        height, width = img_shape[:2]
        pad_diff = height - width

        if pad_diff < 0:
            new_shape = (width, width, 3) if is_rgb else (width, width)
            dst_img = np.zeros(new_shape, dtype=np.uint8)
            dst_img[-pad_diff//2 : pad_diff //2, :, ...] = img
        else:
            new_shape = (height, height, 3) if is_rgb else (height, height)
            dst_img = np.zeros(new_shape, dtype=np.uint8)
            dst_img[:, pad_diff // 2 : -pad_diff // 2, ...] = img

        dst_img = cv2.resize(dst_img, dst_shape)
        return dst_img

    @classmethod
    def expand_mask(cls, img, num_classes):
        mask = np.zeros((*img.shape, num_classes), dtype=np.uint8)
        for i in range(num_classes):
            mask[img == (i + 1), i] = 255
        return mask

    @classmethod
    def collapse_mask(cls, mask, num_classes):
        img = np.zeros(mask.shape[:2], dtype=np.uint8)
        for i in range(num_classes):
            nz_pixels = np.nonzero(mask[:,:, i])
            img[nz_pixels] = i + 1
        return img

    @classmethod
    def read_mask(cls, path, dst_shape, num_classes):
        mask = io.imread(path, plugin='simpleitk').squeeze()
        mask = cls.expand_mask(mask, num_classes)
        mask = cls.resize(mask, dst_shape)
        mask = cls.threshold(mask, 0.43)
        return mask

    @classmethod
    def read_image(cls, path, dst_shape):
        img = io.imread(path, plugin='simpleitk').squeeze()
        img = cls.resize(img, dst_shape)
        img = np.concatenate(3 * [np.expand_dims(img, 2)], axis=2)
        return img

    @classmethod
    def dilate_rgb(cls, img):
        structing_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        dilated_img = cv2.dilate(img, structing_element)
        dilated_img = cls.handle_channel_conficts(dilated_img)
        return dilated_img

    @classmethod
    def erode_rgb(cls, img):
        structing_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        dilated_img = cv2.erode(img, structing_element)
        dilated_img = cls.handle_channel_conficts(dilated_img)
        return dilated_img

    @classmethod
    def handle_channel_conficts(cls, img):
        green = (0, 255, 0)
        blue = (0, 0, 255)
        red = (255, 0, 0)
        r_mask, g_mask, b_mask = img[:, :, 0] == 255, img[:, :, 1] == 255, img[:, :, 2] == 255
        white_mask = b_mask & g_mask & r_mask
        yellow_mask = r_mask & g_mask
        cyan_mask = g_mask & b_mask
        magenta_mask = b_mask & r_mask

        img[yellow_mask] = green
        img[white_mask] = green
        img[cyan_mask] = green
        img[magenta_mask] = red
        return img

    @classmethod
    def _threshold(cls, img, thresh):
        img[img > int(thresh * 255)] = 255
        img[img < 255] = 0
        return img

    @classmethod
    def threshold(cls, img, thresh=0.3):
        for i in range(3):
            img[:, :, i] = cls._threshold(img[:, :, i], thresh)
        img = cls.handle_channel_conficts(img)
        return img

    @classmethod
    def postprocess_batch(cls, images):
        for i, img in enumerate(images):
            images[i] = cls.postprocess(img)
        return images



    @classmethod
    def postprocess(cls, img):
        img = cls.threshold(img)
        img = cls.keep_largest(img)
        img = cls.dilate_rgb(img)
        img = cls.imfill(img)
        img = cls.erode_rgb(img)
        return img

    @classmethod
    def keep_largest(cls, rgb_img):
        dst_rgb_img = np.zeros(rgb_img.shape, dtype=rgb_img.dtype)
        dst_rgb_img[:, :, 1] = rgb_img[:, :, 1]
        for i in [0, 2]:
            img = rgb_img[:, :, i]
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, 4)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            output[output != largest_label] = 0
            dst_rgb_img[:, :, i][np.nonzero(output)] = 255

        return dst_rgb_img

    @classmethod
    def imfill(cls, img, channel_to_fill=1):

        im_th = (img.any(axis=2) * 255).astype(np.uint8)
        # Copy the thresholded image.
        im_floodfill = im_th.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        fill_area = (~ im_th) & im_floodfill_inv

        img[:, :, channel_to_fill][np.nonzero(fill_area)] = 255
        return img



if __name__ == '__main__':
    patient_num = '0053'
    patient_str = f'patient{patient_num}'
    channel_str = '4CH_ED'

    test_image_dir = Path(__file__).joinpath('..', '..', 'data', 'prediction', 'val', patient_str).resolve()
    mask_gt = cv2.imread(str(test_image_dir.joinpath(f'{patient_str}_{channel_str}_gt.png')))
    mask_pred = cv2.imread(str(test_image_dir.joinpath(f'{patient_str}_{channel_str}_pred.png')))
    tm = ImgOps.postprocess(mask_pred.copy())
    imshow_img_pair(mask_gt, mask_pred)

