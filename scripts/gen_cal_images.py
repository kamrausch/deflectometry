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

    if "ver" in orientation:
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

def gen_deflectometry_images(basepath):
    roi = {}
    roi["ver"] = [1640, 2460]
    roi["hor"] = [520, 1390]
    spacing = 20
    for orientation in ["hor", "ver"]:
        offsets = range(roi[orientation][0], roi[orientation][1]+spacing, spacing)
        for offset in offsets:
            filesave_tmp = os.path.join(basepath, f"di_{orientation}_{offset}.png")
            line(filesave_tmp,
                 width=1,
                 resolution=[2160, 3840],
                 offset_in_pixels=offset,
                 plot=False,
                 color="green",
                 orientation=orientation)

    # generate flatfield image
    flatfield = np.zeros((2160, 3840, 3)).astype(np.uint8)
    cv2.imwrite(os.path.join(basepath, f"di_dark.png"), flatfield)
    flatfield[roi["hor"][0]:roi["hor"][1], roi["ver"][0]:roi["ver"][1], 1] = 255
    cv2.imwrite(os.path.join(basepath, f"di_flatfield_green.png"), flatfield)


def gen_many_lines(filesave,
                   width=1,
                   resolution=[2160, 3840],
                   offset_in_pixels=0,
                   plot=False,
                   color="green",
                   orientation="hor"):
    if "ver" in orientation:
        start = 1680
        end = 2700
        num_pts = (end-start)/10 + 1
    else:
        start = 50
        end = 1380
        num_pts = (end-start)/10 + 1        
    offsets = np.linspace(start, end, num_pts)

    for offset in offsets:
        filesave_tmp = f"{filesave}_{offset}.png"
        line(filesave_tmp,
             width=1,
             resolution=[2160, 3840],
             offset_in_pixels=offset.astype(np.uint16),
             plot=plot,
             color=color,
             orientation=orientation)


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


def focusing_target(filesave,
                    width=2,
                    resolution=[2160, 3840],
                    plot=False,
                    color="green",
                    orientation="horizontal"):
    bars(filesave,
         width=width,
         period_in_pixels=2*width,
         resolution=resolution,
         offset_in_pixels=0,
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
