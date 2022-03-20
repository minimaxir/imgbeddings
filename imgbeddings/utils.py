from PIL import Image

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
