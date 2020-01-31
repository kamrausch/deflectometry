import fire
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import cv2
import os
import pdb


COLORS_TO_VALUE = {"red": [255, 0, 0], "green": [0, 255, 0], "blue": [0, 0, 255], "yellow": [255, 255, 0], "white": [255, 255, 255], "black": [0, 0, 0]}

def checkerboard(filesave,
                 block_size=100,               # size of each block in the checkerboard pattern
                 resolution=[2160, 3840],      # resolution of generated image
                 offset=[0, 0],                # offset of the pattern
                 plot=False,                # option to set whether or not to plot image.
                 color="green"):
    """ Generates and saves image which can be displayed on HMD"""

    try:
        value = COLORS_TO_VALUE[color]
    except KeyError:
        raise KeyError(f"Could not find {color} in dictionary list")

    image = np.zeros([2 * block_size, 2 * block_size, 3], dtype='uint8')

    image[0:block_size, 0:block_size, :] = value
    image[block_size:2*block_size, block_size:2*block_size, :] = value

    numTiles = [int(np.ceil(x / (2 * block_size))) for x in resolution]

    image = np.tile(image, (numTiles[0], numTiles[1], 1))
    image = image[0 + offset[0]:resolution[0] + offset[0], 0 + offset[1]:resolution[1] + offset[1], :]

    if plot:
        plt.imshow(image)
        plt.show()

    if filesave:
        cv2.imwrite(filesave, image)
    return image


def bars(filesave,
         width=1,
         period_in_pixels=100,
         resolution=[2160, 3840],
         offset_in_pixels=0,
         plot=False,
         color="white",
         orientation="horizontal"):

    try:
        value = COLORS_TO_VALUE[color]
    except KeyError:
        raise KeyError(f"Could not find {color} in dictionary list")

    if "vertical" in orientation:
        image = np.zeros([resolution[0], period_in_pixels, 3], dtype='uint8')
        image[:, offset_in_pixels:offset_in_pixels+width, :] = value
        numTiles = resolution[1] // period_in_pixels + 1
        image = np.tile(image, (1, numTiles, 1))
        image = image[:, :resolution[1], :]
    else:
        image = np.zeros([period_in_pixels, resolution[1], 3], dtype='uint8')
        image[offset_in_pixels:offset_in_pixels+width, :, :] = value
        numTiles = resolution[0] // period_in_pixels + 1
        image = np.tile(image, (numTiles, 1, 1))
        image = image[:resolution[0], :, :]

    if plot:
        plt.imshow(image)
        plt.show()

    if filesave:
        cv2.imwrite(filesave, image)
    return image


def line(filesave,
         width=1,
         resolution=[2160, 3840],
         offset_in_pixels=0,
         plot=False,
         color="white",
         orientation="horizontal"):

    try:
        value = COLORS_TO_VALUE[color]
    except KeyError:
        raise KeyError(f"Could not find {color} in dictionary list")

    if "vertical" in orientation:
        print(offset_in_pixels)
        image = np.zeros([resolution[0], resolution[1], 3], dtype='uint8')
        image[:, offset_in_pixels:offset_in_pixels+width, :] = value
    else:
        image = np.zeros([resolution[0], resolution[1], 3], dtype='uint8')
        image[offset_in_pixels:offset_in_pixels+width, :, :] = value

    if plot:
        plt.imshow(image)
        plt.show()

    if filesave:
        cv2.imwrite(filesave, image)
    return image


def gen_many_lines(filesave,
                   width=1,
                   resolution=[2160, 3840],
                   offset_in_pixels=0,
                   plot=False,
                   color="white",
                   orientation="vertical"):
    if "ver" in orientation:
        offsets = np.linspace(1000, 2000, 101)

    for offset in offsets:
        filesave_tmp = f"{filesave}_{offset}.png"
        line(filesave_tmp,
             width=1,
             resolution=[2160, 3840],
             offset_in_pixels=offset.astype(np.uint16),
             plot=False,
             color="green",
             orientation="vertical")


def gen_many_bar_images(filesave,
                        width=1,
                        period_in_pixels=100,
                        resolution=[2160, 3840],
                        offset_in_pixels=0,
                        plot=False,
                        color="green",
                        orientation="horizontal"):
    offsets = np.linspace(0, 90, 10)
    for offset in offsets:
        filesave_tmp = f"{filesave}_{offset}.png"
        bars(filesave_tmp,
             width=width,
             period_in_pixels=period_in_pixels,
             resolution=resolution,
             offset_in_pixels=offset.astype(np.uint8),
             color=color,
             orientation=orientation)

def sinusoidal(filesave,
               period_in_pixels=100,
               resolution=[2160, 3840],
               offset_phase=0,
               plot=False,
               color="green",
               orientation="horizontal"):

    try:
        value = COLORS_TO_VALUE[color]
    except KeyError:
        raise KeyError(f"Could not find {color} in dictionary list")

    if "vertical" in orientation:
        resolution[1], resolution[0] = resolution[0], resolution[1]

    num_periods = resolution[1] / period_in_pixels
    image = np.linspace(offset_phase, offset_phase + 2*np.pi*num_periods, resolution[1])
    image = np.matlib.repmat(image, resolution[0], 1)

    image = np.sin(image)
    image = image-np.min(image)
    image = image/np.max(image)

    if "vertical" in orientation:
        image = np.transpose(image)

    image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
    image = image*value
    image = np.round(image).astype(np.uint8)

    if plot:
        plt.imshow(image)
        plt.show()

    if filesave:
        cv2.imwrite(filesave, image)
    return image


if __name__ == '__main__':
    fire.Fire()
