import os
from datetime import datetime
import numpy as np
import fire
from glob import glob
from python_utils.matrox_control import Camera
from python_utils.image_display import ImageDisplay
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import griddata
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter
import shutil
import progressbar
from time import sleep


CROP_IMAGE = False
x_crop = 2000
ROTATE_IMAGE = False


class Deflectometry():

    def __init__(self,
                 serial_number="default",
                 display_filenames_path=r"C:\Users\kameronr\Jobs\python\deflectometry\images",
                 results_path=r"C:\Users\kameronr\Jobs\python\deflectometry\results",
                 archive_path=None):

        self.display_filenames_path = display_filenames_path
        self.results_path = results_path
        self.archive_path = archive_path

        if not os.path.exists(self.display_filenames_path):
            if os.path.exists(r"C:\Users\localuser\Jobs\python\deflectometry"):
                self.display_filenames_path = r"C:\Users\localuser\Jobs\python\deflectometry\display_images"
                self.results_path = r"C:\Users\localuser\Jobs\python\deflectometry\results"
            if os.path.exists(r"C:\Users\tester\Jobs\python\deflectometry"):
                self.display_filenames_path = r"C:\Users\tester\Jobs\python\deflectometry\display_images"
                self.results_path = r"C:\Users\tester\Jobs\python\deflectometry\results"
            else:
                raise Exception(f"Not able to find the paths for the display images")

        if self.archive_path is not None:
            base_dirname = os.path.basename(self.archive_path)
            parts = base_dirname.split("_")
            self.serial_number = "_".join(parts[0: -2])
        else:
            self.serial_number = serial_number

        self.timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"

        self.results_path = os.path.join(self.results_path, f"{self.serial_number}_{self.timestamp}")
        os.makedirs(self.results_path, exist_ok=True)

    def capture_image(self, cam_object, min_DN, max_DN, save_fname=None, dark_image=None, max_retries=5):
        num_tries = 0
        image_invalid = True
        # while image_invalid and num_tries < max_retries:
        image = cam_object.capture_image()
        # if image.is_image_valid(min_DN=min_DN, max_DN=max_DN, roi_size=1000):
                # image_invalid = False
        if save_fname is not None:
            image.save(savename=save_fname, save_yaml=False)
        if dark_image is not None:
            image.subtract_background(background_image=dark_image)
        return image
        #     else:
        #         num_tries += 1
        # else:
        #     raise Exception(f"Unable to capture valid image for {save_fname}")

    def focus_camera(self, bar_target_filename=None):

        if bar_target_filename is None:
            pname = os.path.dirname(self.display_filenames_path)
            bar_target_filename = os.path.join(pname, "focusing_bar_chart_2.png")
        self.image_display = ImageDisplay()
        if not os.path.exists(bar_target_filename):
            raise FileNotFoundError(f"{bar_target_filename} not found")

        filenames = glob(os.path.join(self.display_filenames_path, "di_*.png"))
        # pull out the dark filename so we can use it for dark subtraction. SHould capture this image first
        dark_filename = [x for x in filenames if "dark" in x][0]

        plt.ion()
        contrast = []
        inds= []
        with Camera() as cam:
            cam.set_analog_gain(analog_gain=4)
            # cam.set_exposure_time_ms(exposure_time_ms=1e3)
            cam.set_exposure_time(exposure_time_ms=25)
            self.image_display.show_image(dark_filename)
            dark_image = self.capture_image(cam_object=cam,
                                            dark_image=None,
                                            min_DN=1,
                                            max_DN=65000,
                                            save_fname=None)

            self.image_display.show_image(bar_target_filename)
            for ind in range(0, 1000): 
                image = self.capture_image(cam_object=cam,
                                           dark_image=dark_image.image,
                                           min_DN=1,
                                           max_DN=65000,
                                           save_fname=None)

                Ny, Nx = image.image.shape
                roi = image.image[2200:2600, 2600:3000]
                contrast_ = self.contrast_map(roi, kernel=20)
                contrast_[contrast_>1] = 1
                contrast_[contrast_<0] = 0
                contrast.append(np.mean(contrast_))
                inds.append(ind)
                plt.plot(inds, contrast)
                plt.pause(0.001)

    def contrast_map(self, image, kernel=10):
        tmp_image = image.copy()
        mask_out = self.mask == 0
        tmp_image[mask_out] = np.max(image)
        min_contrast = minimum_filter(tmp_image, (kernel, kernel)).astype(np.float32)
        tmp_image[mask_out] = np.min(image)
        max_contrast = maximum_filter(tmp_image, (kernel, kernel)).astype(np.float32)
        contrast = (max_contrast-min_contrast) / (max_contrast+min_contrast)
        # contrast = gaussian_filter(contrast, sigma=4)
        contrast = contrast * self.mask
        contrast[contrast>1] = 1
        contrast[contrast<0] = 0
        return contrast

    def align_part(self):
        self.image_display = ImageDisplay()
        flatfield_path = r"C:\Users\kameronr\Jobs\python\deflectometry\images\di_flatfield_green.png"
        self.image_display.show_image(flatfield_path)
        breakpoint()
        plt.ion()
        with Camera() as cam:
            while True:
                cam.set_analog_gain(analog_gain=4)
                cam.set_exposure_time(exposure_time_ms=80)
                image = cam.capture_image()
                plt.imshow(image.image)
                breakpoint()
                sleep(2)
                print("it")

        plt.ioff()

    def set_exposure(self):
        self.image_display = ImageDisplay()
        flatfield_path = r"C:\Users\kameronr\Jobs\python\deflectometry\images\di_flatfield_green.png"
        self.image_display.show_image(flatfield_path)
        breakpoint()

    def capture(self):

        exposure_time_ms = 15
        # itialize image display so we can display images on the second monitor
        self.image_display = ImageDisplay()

        filenames = glob(os.path.join(self.display_filenames_path, "di_*.png"))

        # pull out the dark filename so we can use it for dark subtraction. SHould capture this image first
        dark_filename = [x for x in filenames if "dark" in x]
        if len(dark_filename) != 1:
            raise Exception(f"Found {len(dark_filename)} files with the word dark in it when only should find one.")
        else:
            dark_filename = dark_filename[0]
        # now remove dark filename from list
        filenames = [x for x in filenames if "dark" not in x]
        num_files = len(filenames)

        bar = progressbar.ProgressBar(maxval=num_files,
                                      widgets=[progressbar.Bar('=', '[', ']'),
                                      ' ',
                                      progressbar.Percentage()])
        bar.start()
        with Camera() as cam:
            cam.set_analog_gain(analog_gain=4)
            cam.set_exposure_time(exposure_time_ms=exposure_time_ms)
            cam.set_roi(width=1808, height=1808, offset_x=2256, offset_y=3254)
            # push dark image to monitor and grab a picture
            self.image_display.show_image(dark_filename, pause_ms=300)
            save_fname = os.path.join(self.results_path, "images")
            os.makedirs(save_fname, exist_ok=True)
            save_fname = os.path.join(save_fname, self.get_savename(dark_filename))
            dark_image = self.capture_image(cam_object=cam,
                                            dark_image=None,
                                            min_DN=1,
                                            max_DN=65000,
                                            save_fname=save_fname)
            # breakpoint()
            for ind, filename in enumerate(filenames):
                bar.update(ind + 1)
                self.image_display.show_image(filename, pause_ms=300)
                save_fname = os.path.join(self.results_path, "images", self.get_savename(filename))
                save_fname = save_fname.replace("di_", "")
                image = self.capture_image(cam_object=cam,
                                           dark_image=dark_image.image,
                                           min_DN=1,
                                           max_DN=65000,
                                           save_fname=save_fname)
                sleep(0.01)
            bar.finish()

    def get_savename(self, filename):
        basename = os.path.basename(filename).replace("di_", "")
        basename, ext = os.path.splitext(basename)
        return f"{self.serial_number}_{basename}_{self.timestamp}{ext}"

    def measure_spread(self, line_dict, roi_wdith=50, threshold=0.3):
        line_width = np.zeros((len(line_dict["y"])))
        for ind, row in enumerate(line_dict["y"]):
            # extract out region around centroid
            breakpoint()
            roi = self.out[row, np.round(line_dict["x"][ind]-roi_wdith).astype(np.uint16):np.round(line_dict["x"][ind]+roi_wdith).astype(np.uint16)]
            
            max_roi = np.max(roi)
            max_ind = np.where(max_roi == roi)[0][0]
            # find the point where the profile equals the FWHM of the peak
            ind_start = np.where(np.min(np.abs(roi[:roi_wdith] - 0.3*max_roi)) == np.abs(roi[:roi_wdith] - 0.3*max_roi))[0][0]
            sub_roi_ = roi[ind_start:max_ind]
            x = line_dict["y"][ind_start:max_ind]
            x0 = np.interp(0.5*max_roi, sub_roi_, x)


            tmp_ = roi.copy()
            tmp_[:max_ind] = 0
            ind_end = np.where(np.min(np.abs(tmp_ - 0.3*max_roi)) == np.abs(tmp_ - 0.3*max_roi))[0][0]
            sub_roi_ = roi[max_ind:ind_end]
            x = line_dict["y"][max_ind:ind_end]
            x1 = np.interp(0.5*max_roi, sub_roi_, x)
            
            line_width[ind] = x1-x0

        line_width = gaussian_filter1d(line_width, sigma=2)
        return gaussian_filter1d(line_width, sigma=2)
        

    def measure_warp(self, line_dict, plot=False):
        warp = {}
        try:
            lsq = np.polyfit(line_dict["y"], line_dict["x"], 1)
        except:
            breakpoint()
        residual = line_dict["x"]-np.polyval(lsq, line_dict["y"])
        if plot:
            rgb = cv2.cvtColor(self.gray.copy(), cv2.COLOR_GRAY2BGR)
            plt.imshow(rgb)
            plt.plot(np.polyval(lsq, line_dict["y"]), line_dict["y"], 'r-', linewidth=0.5)
            plt.show()
        warp["y"] = line_dict["y"]
        warp["x"] = np.polyval(lsq, line_dict["y"])
        warp["residual"] = residual
        return warp

    def find_line_in_image(self, image, threshold=15, kernel_size=5, plot=True):
        min_contour_area = 500
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if plot:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        image = cv2.bilateralFilter(image, 11, 17, 17)

        retval, thresh = cv2.threshold(image.copy(), threshold, 255, cv2.THRESH_BINARY)

        thresh = thresh.astype(np.uint8)

        thresh = cv2.dilate(thresh, kernel, iterations=5)
        thresh = cv2.erode(thresh, kernel, iterations=5)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours) < 1:
            return {"y": [], "x": []}
        else:
            contours = contours[0]
        if cv2.contourArea(contours) < min_contour_area:
            return {"y": [], "x": []}

        mask = np.zeros_like(image)  # Create mask where white is what we want, black otherwise
        # drawContours requires an array of arrays as input
        cv2.drawContours(mask, [contours], -1, 255, -1)  # Draw filled contour in mask
        self.out = np.zeros_like(image)  # Extract out the object and place into output image
        self.out[mask == 255] = image[mask == 255]

        # with a vertical line in hand, lets calculate the centroid over each row
        Ny, Nx = self.out.shape
        centroids = np.zeros(Ny)
        rows = np.arange(0, Ny)
        for row in rows:
            if np.sum(self.out[row, :]) == 0:
                continue

            centroids[row] = self.centroid_1D(self.out[row, :])

        ind = np.where(centroids > 0)
        rows = rows[ind]
        centroids = centroids[ind]
        # lets lop off a few more just to make sure we don't have edge effects
        # rows = rows[3:-3]
        # centroids = centroids[3:-3]
        # now lets fill in the gaps incase there were some
        try:
            new_rows = np.arange(np.min(rows), np.max(rows)+1)
        except:
            breakpoint()
        centroids - np.interp(new_rows, rows, centroids)
        centroids = gaussian_filter1d(centroids, sigma=2)
        
        return {"y": new_rows, "x": centroids}

    def extract_line(self, gray, line_ind):
        breakpoint()

    def centroid_1D(self, data):
        x = np.arange(0, len(data))
        return np.sum(x*data)/np.sum(data)

    def load_image(self, filename, dark_frame=None):
        if not os.path.exists(filename):
            print(f"{filename} does not exist")

        image = cv2.imread(filename, 0).astype(np.float32)

        if dark_frame is not None:
            image = image - dark_frame

        if ROTATE_IMAGE:
            image = np.rot90(image, 1)

        if CROP_IMAGE:
            image = self.crop_image(image.copy())
        return image

    def copy_images_to_results_path(self):
        _ = shutil.copytree(os.path.join(self.archive_path, "images"), os.path.join(self.results_path, "images"))

    def crop_image(self, image):
        return image[:, :x_crop]

    def get_slopes(self, orientation):
        if "ver" in orientation:
            orientation = "ver"
        else:
            orientation = "hor"
        filenames = glob(os.path.join(self.results_path, "images", f"*{orientation}*.png"))
        filenames = [x for x in filenames if "contrast_bars" not in x]

        x = []
        y = []
        z = []
        spread = []
        for ind, filename in enumerate(filenames):
            # print(f"{filename}: {ind}")
            if "flatfield" in filename:
                continue
            image = self.load_image(filename)

            # apply the mask to filter out background
            image = image * self.mask
            if "hor" in orientation:
                image = np.transpose(image)

            line_dict = self.find_line_in_image(image=image)
            if len(line_dict["y"]) < 10:
                continue
            warp = self.measure_warp(line_dict)
            x.extend(warp["x"])
            y.extend(warp["y"])
            z.extend(warp["residual"])

        if "hor" in orientation:
            x,y = y,x
        return x, y, z
        

    def combine_images(self):
        filenames = glob(os.path.join(self.results_path, "images", "*.png"))
        for ind, filename in enumerate(filenames):
            if "flatfield" in filename:
                continue
            if ind == 0:
                image = self.load_image(filename).astype(np.float32)
            else:
                image = image + self.load_image(filename).astype(np.float32)
        breakpoint()
        plt.imshow(image*self.mask)
        # plt.axis("equal")
        plt.show()            


    def find_mask(self, threshold=100):
        flatfield_filename = os.path.join(self.results_path, "images", "*flatfield*.png")
        filename = glob(flatfield_filename)
        if len(filename) == 0:
            print(f"{flatfield_image} not found")
        elif len(filename) > 1:
            print(f"More than 1 flatfield image was found")
        else:
            filename = filename[0]

        image = self.load_image(filename)
        
        self.mask = ((image > threshold)*1).astype(np.uint8)
        kernel = np.ones((7,7), np.uint8)
        self.mask = cv2.erode(self.mask, kernel, iterations=5)
        self.mask_edge = cv2.Canny(self.mask, 0, 1)

    def generate_slope_map(self, vert_data, horiz_data, plot=False):
        gridPts = 500
        mask_y, mask_x = np.where(self.mask > 0)
        x_min = np.min(mask_x)
        x_max = np.max(mask_x)
        y_min = np.min(mask_y)
        y_max = np.max(mask_y)
        xi = np.linspace(x_min, x_max, gridPts)
        yi = np.linspace(y_min, y_max, gridPts)
        grid_x, grid_y = np.meshgrid(xi, yi)
        grid_vert = griddata(np.transpose(np.array(vert_data[:2])), np.array(vert_data[2]), (grid_x, grid_y), method="linear" )
        grid_horiz = griddata(np.transpose(np.array(horiz_data[:2])), np.array(horiz_data[2]), (grid_x, grid_y), method="linear" )
        slope_map = np.sqrt(grid_vert**2 + grid_horiz**2)*0.136 / (43*25.4) * 1000
        plt.imshow(slope_map)
        plt.colorbar()
        plt.title("Slope Map [mrad]")
        plt.savefig(os.path.join(self.results_path, "slope_map.png"))
        if plot:
            plt.show()
        plt.close()

    def generate_contrast_maps(self, widths=[1, 2, 3, 4], plot=False, normalize=[0.212, 0.835, 1, 1]):
        for ind, width in enumerate(widths):
            filename = glob(os.path.join(self.results_path, "images", f"*contrast_bars*hor_{width}*.png"))
            if len(filename) != 1:
                raise Exception(f"Couldn't find or found multiple filenames")
            else:
                filename = filename[0]
            image = self.load_image(filename, dark_frame=self.dark_image)
            contrast_hor = self.contrast_map(image, kernel=10)

            filename = glob(os.path.join(self.results_path, "images", f"*contrast_bars*ver_{width}*.png"))
            if len(filename) != 1:
                raise Exception(f"Couldn't find or found multiple filenames")
            else:
                filename = filename[0]
            image = self.load_image(filename)
            contrast_ver = self.contrast_map(image, kernel=10)

            contrast = np.sqrt(contrast_ver**2 + contrast_hor**2) * self.mask
            contrast = contrast / normalize[ind]
            contrast[contrast>1] = 1
            contrast[contrast<0] = 0

            mask_y, mask_x = np.where(self.mask > 0)
            x_min = np.min(mask_x)
            x_max = np.max(mask_x)
            y_min = np.min(mask_y)
            y_max = np.max(mask_y)
            contrast = contrast[y_min:y_max, x_min:x_max]

            Ny, Nx = contrast.shape
            roi = contrast[Ny//2-50:Ny//2+50, Nx//2-50:Nx//2+50]

            plt.imshow(contrast, clim=[0, 1])
            plt.colorbar()
            plt.title(f"Contrast Map - Median Contrast = {np.median(roi):.3f}")
            plt.savefig(os.path.join(self.results_path, f"contrast_map_period_{width}.png"))
            if plot:
                plt.show()
            plt.close()


    def get_dark_map(self):
        filename = glob(os.path.join(self.results_path, "images", "*dark*.png"))
        if len(filename) != 1:
            raise Exception(f"Couldn't find or found multiple filenames")
        else:
            filename = filename[0]
        self.dark_image = self.load_image(filename)

    def run_test(self):
        if not self.archive_path:
            self.capture()
        else:
            self.copy_images_to_results_path()

        self.find_mask()

        self.get_dark_map()

        vert_data = self.get_slopes(orientation="ver")
        horiz_data = self.get_slopes(orientation="hor")
        self.generate_slope_map(vert_data=vert_data, horiz_data=horiz_data, plot=False)

        self.generate_contrast_maps()
        # self.combine_images()





if __name__ == '__main__':
    fire.Fire(Deflectometry)
