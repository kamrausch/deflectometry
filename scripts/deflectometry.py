import os
from datetime import datetime
import numpy as np
import fire
from glob import glob
import flir_control as flir
from image_display import ImageDisplay


class Deflectometry():

    def __init__(self,
                 exposure_ms=15,
                 horizontal_resolution=3840,
                 display_filenames_path=r"C:\Users\localuser\Jobs\python\deflectometry\display_images",
                 results_path=r"C:\Users\localuser\Jobs\python\deflectometry\results"):

        self.exposure_ms = exposure_ms
        self.display_filenames_path = display_filenames_path
        self.results_path = results_path

        # itialize image display so we can display images on the second monitor
        self.image_display = ImageDisplay()
        self.image_display.move_to_monitor(horizontal_resolution=horizontal_resolution)

    def capture_image(self, cam_object, min_DN, max_DN, save_fname=None, dark_image=None, max_retries=5):
        num_tries = 0
        image_invalid = True
        while image_invalid and num_tries < max_retries:
            image = cam_object.capture_image()
            if dark_image is not None:
                image.subtract_background(background_image=dark_image)
            if image.is_image_valid(min_DN=50, max_DN=1000, roi_size=50):
                image_invalid = False
                if save_fname is not None:
                    cam_object.save(savename=save_fname)

            else:
                num_tries += 1
        return image

    def run_test(self):
        filenames = glob(os.path.join(self.display_filenames, "di_*.png"))

        # pull out the dark filename so we can use it for dark subtraction. SHould capture this image first
        dark_filename = [x for x in filenames if "dark" in x]
        # now remove dark filename from list
        filenames = [x for x in filenames if "dark" not in x]

        flir_cam = flir.FlirControl()
        with flir_cam:
            flir_cam.set_exposure_time()

            # push dark image to monitor and grab a picture
            self.image_display.show_image(dark_filename)
            save_fname = os.path.join(self.results_path, os.path.basename(dark_filename).replace("de_", ""))
            dark_image = self.capture_image(cam_object=flir_cam,
                                            dark_image=None,
                                            min_DN=50,
                                            max_DN=1000,
                                            save_fname=save_fname)

            for filename in filenames:
                self.image_display.show_image(filename)
                save_fname = os.path.join(self.results_path, os.path.basename(dark_filename).replace("de_", ""))
                image = self.capture_image(cam_object=flir_cam,
                                           dark_image=dark_image,
                                           min_DN=100,
                                           max_DN=5000,
                                           save_fname=save_fname)

                self.measure_defect(image)

    def measure_defect(self, gray, threshold=30, kernel_size=5, plot=True):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        gray = cv2.imread(r"C:\Users\localuser\Jobs\python\deflectometry\results\defect2.png", 0)

        if plot:
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        retval, thresh = cv2.threshold(gray.copy(), threshold, 255, cv2.THRESH_BINARY)

        gray = cv2.dilate(gray, kernel, iterations=1)
        gray = cv2.erode(gray, kernel, iterations=1)

        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

        # https://stackoverflow.com/questions/28759253/how-to-crop-the-internal-area-of-a-contour


if __name__ == '__main__':
    fire.Fire(Deflectometry)
