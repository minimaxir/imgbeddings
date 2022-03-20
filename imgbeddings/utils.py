from random import uniform, shuffle

from PIL import Image, ImageEnhance
import numpy as np

# https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/


def square_pad(pil_img, background_color=(0, 0, 0)):
    """Pads a PIL image to a square, before augmentation."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def symmetric_img_aug(img, r_shift, r_degrees, buffer, background_color):

    # apply color transformations
    funcs = [ImageEnhance.Color, ImageEnhance.Contrast, ImageEnhance.Brightness]
    shuffle(funcs)

    for func in funcs:
        img = func(img).enhance(uniform(1.0 - r_shift, 1.0 + r_shift))

    if r_degrees > 0:
        # apply padding before rotation
        width, height = img.size

        width_b = int(width * buffer)
        height_b = int(height * buffer)
        img_b = Image.new(img.mode, (width_b, height_b), background_color)
        offset = ((width_b - width) // 2, (height_b - height) // 2)
        img_b.paste(img, offset)
        img_b = img_b.rotate(uniform(0.0 - r_degrees, 0.0 + r_degrees))

        # identify what areas do not match the background color
        # to see where to crop
        img_b_array = np.array(img_b)
        nonbg_array = (img_b_array != background_color).all(axis=2)
        idx = np.where(nonbg_array)

        left_crop = np.min(idx[1])
        right_crop = np.max(idx[1])
        top_crop = np.min(idx[0])
        bottom_crop = np.max(idx[0])
        img_b = img_b.crop((left_crop, top_crop, right_crop, bottom_crop))

        return img_b
    return img
