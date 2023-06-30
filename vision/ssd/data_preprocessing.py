from ..transforms.transforms import *


class ScaleByStd:
    def __init__(self, std: float):
        self.std = std
 
    def __call__(self, img, boxes=None, labels=None):
        return (img / self.std, boxes, labels)


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(), # np.float32 に変換するクラス
            # PhotometricDistort(), # 測光のゆがみを入れるクラス
            Expand(self.mean), # 拡大・縮小（平均から行う）クラス
            RandomSampleCrop(), # ランダムにクロッピングを行うクラス
            RandomMirror(), # 左右反転クラス
            ToPercentCoords(), # 座標のrangeを0-255から0-1に変換するクラス
            Resize(self.size), # 画像サイズを変更するクラス
            # 画像のタイプをnp.float32 -> np.float32に変換し，
            # 全画素からstdを引いた値にするクラス
            SubtractMeans(self.mean),
            #lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ScaleByStd(std), # stdで画素を割るクラス
            ToTensor(), # np.float32からテンソルに変換するクラス
        ])
        
    def __call__(self, img, boxes, labels):
        """
        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            #lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ScaleByStd(std),
            ToTensor(),
        ])
        
    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            #lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ScaleByStd(std),
            ToTensor()
        ])
        
    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image