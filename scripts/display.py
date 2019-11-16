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

    def generate_canvas(self):
        monitor = self.monitors[self.monitor_ind]
        self.canvas = tk.Canvas(self.window, bg="blue", width=monitor.width, height=monitor.height, highlightthickness=0)
        self.image = np.zeros((monitor.height, monitor.width, 3)).astype(np.uint8)
        self.image[:, :, 2] = 255
        tkimage = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.image.copy()))
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

        self.generate_canvas()

    def show_image(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} does not exist")
        self.image = PIL.ImageTk.PhotoImage(PIL.Image.open(filename))

        self.canvas.itemconfig(self.image_on_canvas, image=self.image)

    def close(self):
        self.window.destroy()

if __name__ == '__main__':
    fire.Fire(ImageViewer)


# iv = ImageViewer()
# iv.move_to_monitor(horizontal_resolution=3840)
# iv.show_image(r"C:\Users\kam_r\Jobs\python\deflectometry\display_images\green.png")
# time.sleep(5)
# iv.show_image(r"C:\Users\kam_r\Jobs\python\deflectometry\display_images\red.png")
# time.sleep(5)
# iv.show_image(r"C:\Users\kam_r\Jobs\python\deflectometry\display_images\green.png")
# time.sleep(5)
# iv.show_image(r"C:\Users\kam_r\Jobs\python\deflectometry\display_images\red.png")
# iv.close()


# breakpoint()