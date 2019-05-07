import argparse
import time

import cv2
import numpy as np
from skimage import transform


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "image",
        help="path to input image file"
    )
    return ap.parse_args()


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def get_max_freq_elem(arr):

    max_arr = []
    freqs = {}
    for i in arr:
        if i in freqs:
            freqs[i] += 1
        else:
            freqs[i] = 1

    sorted_keys = sorted(freqs, key=freqs.get, reverse=True)
    max_freq = freqs[sorted_keys[0]]

    for k in sorted_keys:
        if freqs[k] == max_freq:
            max_arr.append(k)

    return max_arr


def compare_sum(value):
    if 44 <= value <= 46:
        return True
    else:
        return False


def calculate_deviation(angle):
    angle_in_degrees = np.abs(angle)
    deviation = np.abs(np.pi / 4 - angle_in_degrees)

    return deviation


def derotate2(image):
    num_peaks = 20

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # t = time.time()
    edges = cv2.Canny(gray, 25, 255)
    # print("canny: {:.3f}ms".format((time.time() - t) * 1000))
    # t = time.time()
    h, a, d = transform.hough_line(edges)
    # print("hough_line: {:.3f}ms".format((time.time() - t) * 1000))
    # t = time.time()
    _, ap, _ = transform.hough_line_peaks(h, a, d, num_peaks=num_peaks)
    # print("hough_line_peaks: {:.3f}ms".format((time.time() - t) * 1000))

    if len(ap) == 0:
        raise RuntimeError("Bad Quality image")

    absolute_deviations = [calculate_deviation(k) for k in ap]
    average_deviation = np.mean(np.rad2deg(absolute_deviations))
    ap_deg = [np.rad2deg(x) for x in ap]

    bin_0_45 = []
    bin_45_90 = []
    bin_0_45n = []
    bin_45_90n = []

    for ang in ap_deg:
        deviation_sum = int(90 - ang + average_deviation)
        if compare_sum(deviation_sum):
            bin_45_90.append(ang)
            continue

        deviation_sum = int(ang + average_deviation)
        if compare_sum(deviation_sum):
            bin_0_45.append(ang)
            continue

        deviation_sum = int(-ang + average_deviation)
        if compare_sum(deviation_sum):
            bin_0_45n.append(ang)
            continue

        deviation_sum = int(90 + ang + average_deviation)
        if compare_sum(deviation_sum):
            bin_45_90n.append(ang)

    angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
    lmax = 0

    for j in range(len(angles)):
        l = len(angles[j])
        if l > lmax:
            lmax = l
            maxi = j

    if lmax:
        ans_arr = get_max_freq_elem(angles[maxi])
        ans_res = np.mean(ans_arr)

    else:
        ans_arr = get_max_freq_elem(ap_deg)
        ans_res = np.mean(ans_arr)

    angle = ans_res

    # if 0 <= angle <= 90:
    #     rot_angle = angle - 90 + r_angle
    if -45 <= angle < 0:
        angle = angle - 90
    if -90 <= angle < -45:
        angle = 90 + angle
    if 90 <= angle <= 180:
        angle -= 90

    rotated = rotate(image, angle)

    return rotated, angle


def derotate(image):
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(
        gray, 200, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # rotate the image to deskew it
    return rotate(image, angle), angle


def rotate(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(m[0, 0])
    abs_sin = abs(m[0, 1])

    # find the new width and height bounds
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    m[0, 2] += bound_w / 2 - center[0]
    m[1, 2] += bound_h / 2 - center[1]

    return cv2.warpAffine(
        image, m, (bound_w, bound_h),
        flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE
    )


def main():
    args = parse_args()
    image = cv2.imread(args.image)

    t = time.time()
    rotated, angle = derotate2(image)
    print("derotate: {:.3f}ms".format((time.time() - t) * 1000))
    print("[INFO] angle: {:.3f}".format(angle))

    if image.shape[0] >= 1000:
        image = image_resize(image, height=1000)
        rotated = image_resize(rotated, height=1000)

    if image.shape[1] >= 1900:
        image = image_resize(image, width=1900)
        rotated = image_resize(rotated, width=1900)

    cv2.imshow("Input", image)
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
