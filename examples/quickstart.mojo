from mojocv.mat import Mat
from mojocv.imgproc import gaussian_blur, otsu, threshold
from mojocv.core import flip


def main() raises:
    # A 64 x 64 grey image: dark left half (40), bright right half (200).
    var img = Mat[DType.uint8](64, 64, 1)
    for r in range(64):
        for c in range(64):
            img.set(r, c, 0, UInt8(40) if c < 32 else UInt8(200))

    var t = otsu(img)  # cv2.THRESH_OTSU
    var mask = threshold(img, Float64(t), 255.0, 0)  # THRESH_BINARY
    var mirrored = flip(mask, 1)  # cv2.flip, around the y axis
    var blurred = gaussian_blur(img, 5, 0.0)  # cv2.GaussianBlur((5, 5), 0)

    print("otsu threshold", t)
    print("mask row 0: left", mask.at(0, 0, 0), "right", mask.at(0, 63, 0))
    print("mirrored row 0: left", mirrored.at(0, 0, 0), "right", mirrored.at(0, 63, 0))
    print("blur across the edge:", blurred.at(32, 29, 0), blurred.at(32, 31, 0), blurred.at(32, 33, 0))

    if mask.at(0, 0, 0) != 0 or mask.at(0, 63, 0) != 255:
        raise Error("quickstart: threshold did not split the halves")
    if mirrored.at(0, 0, 0) != 255 or mirrored.at(0, 63, 0) != 0:
        raise Error("quickstart: flip did not mirror the mask")
    if blurred.at(32, 31, 0) <= 40 or blurred.at(32, 31, 0) >= 200:
        raise Error("quickstart: blur did not soften the edge")
