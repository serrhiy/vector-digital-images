import os

import cv2

RESOURCES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "resources")
)


def main():
    kpi_image_path = os.path.join(RESOURCES_DIR, "kpi.jpg")
    original_image = cv2.imread(kpi_image_path, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blured, 10, 500)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for contour in contours:
        perimiter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimiter, True)
        if len(approx) == 4:
            rectangles.append(approx)
    cv2.drawContours(original_image, rectangles, -1, (0, 255, 0), 1)

    cv2.imshow("KPI", original_image)
    cv2.imshow("KPI Processed", edged)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
