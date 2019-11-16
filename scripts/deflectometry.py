import os
from datetime import datetime
import numpy as np
import fire
import pyglet
import flir_control
import 


class ImageWindow(pyglet.window.Window):

    def __init__(self):
        display = pyglet.canvas.get_display()
        screens = display.get_screens()
        screen = screens[0]
        self.window = pyglet.window.Window.__init__(self, style=pyglet.window.Window.WINDOW_STYLE_BORDERLESS, screen=screen, fullscreen=True)

    def draw_image(self, image_name):
        image = pyglet.image.load(image_name)
        self.image = pyglet.sprite.Sprite(image)
        pyglet.app.run()

    def on_draw(self):
        # self.window.clear()
        self.image.draw()

    def on_key_press(self, symbol, modifiers):
        self.window.close()

    def get_screen_from_resolution(self, screens, resolution):
        if len(screens) < 2:
            raise Exception(f"Only {len(screens)} found.")

        for screen in screens:
            if screen.width == resolution[0] and screen.height == resolution[1]:
                break
        else:
            raise Exception(f"A monitor was not found with the specified resolution")

        return screen


    def center_image(self, image):
        """Sets an image's anchor point to its center"""
        image.anchor_x = image.width // 2
        image.anchor_y = image.height // 2


    def show_images(self):
        self.get_fullscreen_window()
        breakpoint()
        self.set_image_dir()


# image_window = ImageWindow()
# image_window.draw_image(r"C:\Users\kam_r\Jobs\python\deflectometry\images\red.png")


# import pyglet
# import sys

# class ImageWindow(pyglet.window.Window):

#     def __init__(self,*args,**kwargs):
#         pyglet.window.Window.__init__(self, *args,**kwargs)

#     def draw_image(self):
#         image = pyglet.image.load(r'C:\Users\kam_r\Jobs\python\deflectometry\images\red.png')
#         self.image_sprite = pyglet.sprite.Sprite(image)
#         pyglet.app.run()

#     def on_draw(self):
#         self.image_sprite.draw()


# image_window = ImageWindow()
# image_window.draw_image()
