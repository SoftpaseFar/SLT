from PIL import Image, ImageEnhance
import random


class Brightness(object):
    def __init__(self, min=1, max=1.5) -> None:  # 调整了 min 和 max 的范围
        self.min = min
        self.max = max

    def __call__(self, clip):
        factor = random.uniform(self.min, self.max)
        new_clip = []
        for img in clip:
            if isinstance(img, Image.Image):
                enh_bri = ImageEnhance.Brightness(img)
                new_img = enh_bri.enhance(factor)
                new_clip.append(new_img)
            else:
                raise TypeError('Expected PIL.Image but got {0}'.format(type(img)))
        return new_clip


class Color(object):
    def __init__(self, min=1, max=1.5) -> None:  # 调整了 min 和 max 的范围
        self.min = min
        self.max = max

    def __call__(self, clip):
        factor = random.uniform(self.min, self.max)
        new_clip = []
        for img in clip:
            if isinstance(img, Image.Image):
                enh_col = ImageEnhance.Color(img)
                new_img = enh_col.enhance(factor)
                new_clip.append(new_img)
            else:
                raise TypeError('Expected PIL.Image but got {0}'.format(type(img)))
        return new_clip
