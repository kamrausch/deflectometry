import os
from datetime import datetime
import numpy as np
import fire
from glob import glob
from python_utils import flir_control as flir
from python_utils.image_display import ImageDisplay
import matplotlib.pyplot as plt
import cv2


class Deflectometry():

    def __init__(self,
                 serial_number="default",
                 exposure_ms=15,
                 horizontal_resolution=3840,
                 display_filenames_path=r"C:\Users\kam_r\Jobs\python\deflectometry\display_images",
                 results_path=r"C:\Users\kam_r\Jobs\python\deflectometry\results"):

        self.serial_number = serial_number
        self.exposure_ms = exposure_ms
        self.display_filenames_path = display_filenames_path
        self.results_path = results_path

        if not os.path.exists(self.display_filenames_path):
            if os.path.exists(r"C:\Users\localuser\Jobs\python\deflectometry"):
                self.display_filenames_path = r"C:\Users\localuser\Jobs\python\deflectometry\display_images"
                self.results_path = r"C:\Users\localuser\Jobs\python\deflectometry\results"
            if os.path.exists(r"C:\Users\tester\Jobs\python\deflectometry"):
                self.display_filenames_path = r"C:\Users\tester\Jobs\python\deflectometry\display_images"
                self.results_path = r"C:\Users\tester\Jobs\python\deflectometry\results"
        else:
            raise Exception(f"Not able to find the paths for the display images")

        self.timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"

        self.results_path = os.path.join(self.results_path, f"{self.serial_number}_{self.timestamp}")
        os.makedirs(self.results_path, exist_ok=True)

        # itialize image display so we can display images on the second monitor
        self.image_display = ImageDisplay()

    def capture_image(self, cam_object, min_DN, max_DN, save_fname=None, dark_image=None, max_retries=5):
        num_tries = 0
        image_invalid = True
        while image_invalid and num_tries < max_retries:
            image = cam_object.capture_image()
            # breakpoint()
            if image.is_image_valid(min_DN=min_DN, max_DN=max_DN):
                image_invalid = False
                if save_fname is not None:
                    image.save(savename=save_fname)
                if dark_image is not None:
                    image.subtract_background(background_image=dark_image)
                return image
            else:
                num_tries += 1
        else:
            raise Exception(f"Unable to capture valid image for {save_fname}")

    def run_test(self):
        filenames = glob(os.path.join(self.display_filenames_path, "di_*.png"))

        # pull out the dark filename so we can use it for dark subtraction. SHould capture this image first
        dark_filename = [x for x in filenames if "dark" in x]
        if len(dark_filename) != 1:
            raise Exception(f"Found {len(dark_filename)} files with the word dark in it when only should find one.")
        else:
            dark_filename = dark_filename[0]
        # now remove dark filename from list
        filenames = [x for x in filenames if "dark" not in x]

        flir_cam = flir.FlirControl()
        with flir_cam:
            flir_cam.set_gain_db(gain_db=0.0)
            # push dark image to monitor and grab a picture
            self.image_display.show_image(dark_filename)
            save_fname = os.path.join(self.results_path, self.get_savename(dark_filename))
            dark_image = self.capture_image(cam_object=flir_cam,
                                            dark_image=None,
                                            min_DN=1,
                                            max_DN=65000,
                                            save_fname=save_fname)
            # breakpoint()
            for filename in filenames:
                if "flatfield" in filename:
                    flir_cam.set_exposure_time(exposure_time_ms=1e3)
                else:
                    flir_cam.set_exposure_time(exposure_time_ms=5e3)
                self.image_display.show_image(filename)
                save_fname = os.path.join(self.results_path, self.get_savename(filename))
                image = self.capture_image(cam_object=flir_cam,
                                           dark_image=dark_image,
                                           min_DN=1,
                                           max_DN=65000,
                                           save_fname=save_fname)

                # self.measure_defect(image)

    def get_savename(self, filename):
        basename = os.path.basename(filename).replace("de_", "")
        basename, ext = os.path.splitext(basename)
        return f"{self.serial_number}_{basename}_{self.timestamp}{ext}"


    def measure_defect(self, gray, threshold=30, kernel_size=5, plot=True):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        gray = cv2.imread(r"C:\Users\localuser\Jobs\python\deflectometry\results\defect.png", 0)

        if plot:
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        retval, thresh = cv2.threshold(gray.copy(), threshold, 255, cv2.THRESH_BINARY)

        gray = cv2.dilate(gray, kernel, iterations=1)
        gray = cv2.erode(gray, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        mask = np.zeros_like(gray)  # Create mask where white is what we want, black otherwise
        cv2.drawContours(mask, contours, 0, 255, -1)  # Draw filled contour in mask
        out = np.zeros_like(gray)  # Extract out the object and place into output image
        out[mask == 255] = gray[mask == 255]

        # https://stackoverflow.com/questions/28759253/how-to-crop-the-internal-area-of-a-contour


if __name__ == '__main__':
    fire.Fire(Deflectometry)
