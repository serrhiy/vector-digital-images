import os

import cv2

RESOURCES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "resources")
)


def main():
    kpi_image_path = os.path.join(RESOURCES_DIR, "kpi.jpg")
    original_image = cv2.imread(kpi_image_path, cv2.IMREAD_GRAYSCALE)

    blured = cv2.GaussianBlur(original_image, (3, 3), 0)
    edged = cv2.Canny(blured, 100, 200)

    cv2.imshow("KPI", original_image)
    cv2.imshow("KPI Processed", edged)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
