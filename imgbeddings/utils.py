from random import uniform, shuffle

from PIL import Image, ImageEnhance
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions


def square_pad(img, background_color=(0, 0, 0)):
    """Pads a PIL image to a square, before augmentation."""
    # https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
    width, height = img.size
    if width == height:
        return img
    if not background_color:
        background_color = get_dominant_color(img)

    if width > height:
        result = Image.new(img.mode, (width, width), background_color)
        result.paste(img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(img.mode, (height, height), background_color)
        result.paste(img, ((height - width) // 2, 0))
        return result


def symmetric_img_aug(
    img,
    r_shift=0.75,
    r_degrees=15,
    background_color=(0, 0, 0),
    expand=True,
    pad_to_square=True,
):

    assert r_shift > 0 or r_degrees > 0, "No augmentation is specified."
    if r_shift > 0:
        # apply color transformations
        funcs = [ImageEnhance.Color, ImageEnhance.Contrast, ImageEnhance.Brightness]
        shuffle(funcs)

        for func in funcs:
            img = func(img).enhance(uniform(1.0 - r_shift, 1.0 + r_shift))

    if r_degrees > 0:
        if not background_color:
            background_color = get_dominant_color(img)
        img = img.rotate(
            uniform(0.0 - r_degrees, 0.0 + r_degrees),
            expand=expand,
            fillcolor=background_color,
        )
    if pad_to_square:
        return square_pad(img, background_color)
    return img


def get_dominant_color(img_input):
    # https://stackoverflow.com/a/61730849
    img = img_input.copy()
    img.convert("RGB")
    img = img.resize((1, 1), resample=0)
    dominant_color = img.getpixel((0, 0))
    return dominant_color


def create_session_for_provider(model_path, provider, gpu):
    if not provider:
        if gpu:
            provider = "CUDAExecutionProvider"
        else:
            provider = "CPUExecutionProvider"
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    options.use_deterministic_compute = True
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()
    return session
