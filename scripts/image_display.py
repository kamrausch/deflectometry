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

Example calls
iv = ImageViewer()
iv.move_to_monitor(horizontal_resolution=3840)
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

        self.generate_canvas()

    def generate_canvas(self):
        monitor = self.monitors[self.monitor_ind]
        self.canvas = tk.Canvas(self.window, bg="blue", width=monitor.width, height=monitor.height, highlightthickness=0)
        self.image = np.zeros((monitor.height, monitor.width, 3)).astype(np.uint8)
        self.image[:, :, 2] = 255
        tkimage = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.image.copy()))
        self.image_on_canvas = self.canvas.create_image(0, 0, image=tkimage, anchor=tk.NW)
        self.canvas.pack()

    def show_image(self, filename, pause_ms=10):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} does not exist")
        self.image = PIL.ImageTk.PhotoImage(PIL.Image.open(filename))

        self.canvas.itemconfig(self.image_on_canvas, image=self.image)
        time.sleep(pause_ms/1000)

    def close(self):
        self.window.destroy()


if __name__ == '__main__':
    fire.Fire(ImageDisplay)
