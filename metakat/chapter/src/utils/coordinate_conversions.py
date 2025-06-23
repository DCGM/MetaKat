# File containing coordiante conversion helper functions.
# Author: Richard Blažo
# File name: coordinate_conversions.py
# Description: This file contains functions for converting coordinates between different formats.

def labelStudioToYOLO(coords):
    x, y, w, h = coords

    w /= 100
    h /= 100
    x /= 100
    y /= 100

    x = x + (w / 2)
    y = y + (h / 2)
    return (x, y, w, h)


def YOLOToLabelStudio(coords):
    x, y, w, h = coords

    x = x - (w / 2)
    y = y - (h / 2)

    w *= 100
    h *= 100
    x *= 100
    y *= 100

    return (x, y, w, h)


def YOLOToOCR(coords, imgShape):
    x, y, w, h = coords
    h_img, w_img = imgShape

    x = x * w_img
    y = y * h_img
    w = w * w_img
    h = h * h_img

    left = int(x - (w / 2))
    right = int(x + (w / 2))
    top = int(y - (h / 2))
    bottom = int(y + (h / 2))

    return (left, top, right, bottom)
