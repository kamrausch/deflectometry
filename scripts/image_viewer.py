import tkinter as tk
import PIL.Image, PIL.ImageTk
import screeninfo
import os
import numpy as np
import time
import fire

class ImageViewer():

    def __init__(self, topmost=True, borderless=True, horizontal_resolution=3840):

        self.horizontal_resolution = horizontal_resolution

        self.window = tk.Tk()
        if borderless:
            self.window.overrideredirect(1)
        if topmost:
            self.window.attributes('-topmost', 1)

        self.monitors = screeninfo.get_monitors()
        self.window.config(cursor="none")

        self.move_to_monitor()
        self.generate_canvas()

    def generate_canvas(self):
        global tkimage
        global image
        monitor = self.monitors[self.monitor_ind]
        self.canvas = tk.Canvas(self.window, bg="blue", width=monitor.width, height=monitor.height, highlightthickness=0)
        image = np.zeros((monitor.height, monitor.width, 3)).astype(np.uint8)
        image[:, :, 2] = 255
        tkimage = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(image))
        self.image_on_canvas = self.canvas.create_image(0, 0, image=tkimage, anchor=tk.NW)
        self.canvas.pack()

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

    def show_image(self, filename):
        global image
        global tkimage
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} does not exist")
        image = PIL.Image.open(filename)
        tkimage = PIL.ImageTk.PhotoImage(image)

        self.canvas.itemconfig(self.image_on_canvas, image=tkimage)
        self.canvas.update()

    def close(self):
        self.window.destroy()

iv = ImageViewer()
# breakpoint()
iv.show_image(r"C:\Users\kam_r\Jobs\python\deflectometry\display_images\di_green.png")
# image.show()
iv.show_image(r"C:\Users\kam_r\Jobs\python\deflectometry\display_images\di_red.png")
# image.show()
iv.show_image(r"C:\Users\kam_r\Jobs\python\deflectometry\display_images\di_green.png")
# image.show()
iv.show_image(r"C:\Users\kam_r\Jobs\python\deflectometry\display_images\di_red.png")
iv.close()


# breakpoint()