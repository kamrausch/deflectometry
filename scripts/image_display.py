import tkinter as tk
import PIL.Image
import PIL.ImageTk
import screeninfo
import os
import numpy as np
import fire
import time

"""
This class puts an image on a monitor specified by the horizontal resolution of that monitor. The default
horitontal resolution the code looks for is 3840. If more than one monitor is found with the specified
resolution, than an exception is thrown. You can overcome this by changing the scale of the monitor in
display settings.

    :param topmost (bool):               Option to have the generated window be the topmost window. Not sure if this
                                            would ever want to be set to False however.
    :param borderless (bool):            Option to turn on a window border so the user can close window directly
    :param hide mouse (bool):            Option to have the mouse cursor hidden while over the window
    :param horizontal_resolution (int):  Option to specify the horizontal resolution of the monitor one wishes to push the image to

Example calls
    from image_display import ImageDisplay
    iv = ImageDisplay(horizontal_resolution=3840)
    iv.show_image(filename1)
    ...
    iv.show_image(filename2)
    iv.close()
"""


class ImageDisplay():

    def __init__(self, topmost=True, borderless=True, hide_mouse=True, horizontal_resolution=3840):

        self.horizontal_resolution = horizontal_resolution

        self.window = tk.Tk()
        if borderless:
            self.window.overrideredirect(1)
        if topmost:
            self.window.attributes('-topmost', 1)

        self.monitors = screeninfo.get_monitors()

        if hide_mouse:
            self.window.config(cursor="none")

        self.move_to_monitor()

    def move_to_monitor(self):
        count = 0
        for ind, monitor in enumerate(self.monitors):
            if self.horizontal_resolution == monitor.width:
                count += 1
                self.monitor_ind = ind

        if count > 1:
            raise Exception(f"Found more than one monitor with a horizontal resolution = {self.horizontal_resolution}")

        if count < 1:
            raise Exception(f"Didnt find a monitor with resolution = {self.horizontal_resolution}")

        monitor = self.monitors[self.monitor_ind]
        self.window.geometry(f"{monitor.width}x{monitor.height}+{monitor.x}+{monitor.y}")
        time.sleep(0.2)  # TODO: make sure this is actually needed. Or maybe time extended

        self.generate_canvas()

    def generate_canvas(self):
        monitor = self.monitors[self.monitor_ind]
        self.canvas = tk.Canvas(self.window, bg="blue", width=monitor.width, height=monitor.height, highlightthickness=0)
        self.image = np.zeros((monitor.height, monitor.width, 3)).astype(np.uint8)
        self.image[:, :, 2] = 255
        self.tkimage = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.image))
        self.image_on_canvas = self.canvas.create_image(0, 0, image=self.tkimage, anchor=tk.NW)
        self.canvas.pack()

    def show_image(self, filename, pause_ms=3):
        # TODO: if image size is less than canvas size, show image centered within the canvas
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} does not exist")
        self.tkimage = PIL.ImageTk.PhotoImage(PIL.Image.open(filename))

        self.canvas.itemconfig(self.image_on_canvas, image=self.tkimage)
        self.canvas.update()
        time.sleep(pause_ms/1000)

    def close(self):
        self.window.destroy()


if __name__ == '__main__':
    from glob import glob
    display_filenames_path = r"C:\Users\kam_r\Jobs\python\deflectometry\display_images"
    filenames = glob(os.path.join(display_filenames_path, "di_*.png"))
    disp = ImageDisplay(horizontal_resolution=3840)
    for filename in filenames:
        print(f"Displaying {filename}")
        disp.show_image(filename)
        time.sleep(2)
    disp.close()
