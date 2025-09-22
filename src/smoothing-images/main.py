import os
import cv2, numpy
from collections.abc import Callable

RESOURCES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "resources")
)


def averaging(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """As simple as possible"""
    DIMENSION = 5
    # kernel = numpy.ones((DIMENSION, DIMENSION), numpy.float32) / (DIMENSION**2)
    # return cv2.filter2D(image, -1, kernel)
    return cv2.blur(image, DIMENSION)


def GaussianBlurring(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Is highly effective in removing Gaussian noise"""
    DIMENSION = 5
    return cv2.GaussianBlur(image, (DIMENSION, DIMENSION), 0)


def medianBlur(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Is highly effective against salt-and-pepper noise"""
    DIMENSION = 3
    return cv2.medianBlur(image, DIMENSION)


def bilateralBlur(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Is highly effective in noise removal while keeping edges sharp"""
    return cv2.bilateralFilter(image, 9, 75, 75)


def main(transform: Callable[[cv2.typing.MatLike], cv2.typing.MatLike]):
    kpi_image_path = os.path.join(RESOURCES_DIR, "kpi.jpg")
    image = cv2.imread(kpi_image_path, cv2.IMREAD_COLOR)
    processed = transform(image)

    cv2.imshow("KPI", image)
    cv2.imshow("KPI Processed", processed)
    cv2.waitKey(0)


if __name__ == "__main__":
    transforms = (averaging, GaussianBlurring, medianBlur, bilateralBlur)

    main(transforms[3])
