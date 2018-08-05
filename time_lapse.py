"""
Module docstring.
"""

import datetime
import logging

import os
import glob

from dateutil.rrule import rrule, MONTHLY

from itertools import compress

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sentinelhub.data_request import WmsRequest, WcsRequest
from sentinelhub.constants import MimeType, CustomUrlParam
from s2cloudless import CloudMaskRequest, MODEL_EVALSCRIPT


LOGGER = logging.getLogger(__name__)


class SentinelHubTimelapse(object):
    """
    Class for creating timelapses with Sentinel-2 images using Sentinel Hub's Python library.
    """

    def __init__(self, project_name, bbox, time_interval, instance_id, full_size=(1920, 1080), preview_size=(455, 256),
                 cloud_mask_res=('60m', '60m'), use_atmcor=True, layer='TRUE_COLOR',
                 time_difference=datetime.timedelta(seconds=-1)):

        self.project_name = project_name
        self.preview_request = WmsRequest(data_folder=project_name + '/previews', layer=layer, bbox=bbox,
                                          time=time_interval, width=preview_size[0], height=preview_size[1],
                                          maxcc=1.0, image_format=MimeType.PNG, instance_id=instance_id,
                                          custom_url_params={CustomUrlParam.TRANSPARENT: True},
                                          time_difference=time_difference)

        self.fullres_request = WmsRequest(data_folder=project_name + '/fullres', layer=layer, bbox=bbox,
                                          time=time_interval, width=full_size[0], height=full_size[1],
                                          maxcc=1.0, image_format=MimeType.PNG, instance_id=instance_id,
                                          custom_url_params={CustomUrlParam.TRANSPARENT: True,
                                              CustomUrlParam.ATMFILTER: 'ATMCOR'} if use_atmcor else {CustomUrlParam.TRANSPARENT: True},
                                          time_difference=time_difference)

        wcs_request = WcsRequest(layer=layer, bbox=bbox, time=time_interval,
                                 resx=cloud_mask_res[0], resy=cloud_mask_res[1], maxcc=1.0,
                                 image_format=MimeType.TIFF_d32f, instance_id=instance_id,
                                 time_difference=time_difference, custom_url_params={CustomUrlParam.EVALSCRIPT:
                                                                                     MODEL_EVALSCRIPT})

        self.cloud_mask_request = CloudMaskRequest(wcs_request)

        self.transparency_data = None
        self.preview_transparency_data = None
        self.invalid_coverage = None

        self.dates = self.preview_request.get_dates()
        if not self.dates:
            raise ValueError('Input parameters are not valid. No Sentinel 2 image is found.')

        if self.dates != self.fullres_request.get_dates():
            raise ValueError('Lists of previews and full resolution images do not match.')

        if self.dates != self.cloud_mask_request.get_dates():
            raise ValueError('List of previews and cloud masks do not match.')

        self.mask = np.zeros((len(self.dates),), dtype=np.uint8)
        self.cloud_masks = None
        self.cloud_coverage = None

        self.full_res_data = None
        self.previews = None
        self.full_size = full_size
        self.timelapse = None

        LOGGER.info('Found %d images of %s between %s and %s.', len(self.dates), project_name,
                    time_interval[0], time_interval[1])

        LOGGER.info('\nI suggest you start by downloading previews first to see,\n'
                    'if BBOX is OK, images are usefull, etc...\n'
                    'Execute get_previews() method on your object.\n')

    def get_previews(self, redownload=False):
        """
        Downloads and returns an numpy array of previews if previews were not already downloaded and saved to disk.
        Set `redownload` to True if to force downloading the previews again.
        """

        self.previews = np.asarray(self.preview_request.get_data(save_data=True, redownload=redownload))
        self.preview_transparency_data = self.previews[:,:,:,-1]

        LOGGER.info('%d previews have been downloaded and stored to numpy array of shape %s.', self.previews.shape[0],
                    self.previews.shape)

    def save_fullres_images(self, redownload=False):
        """
        Downloads and saves fullres images used to produce the timelapse. Note that images for all available dates
        within the specified time interval are downloaded, although they will be for example masked due to too high
        cloud coverage.
        """
        
        data4d = np.asarray(self.fullres_request.get_data(save_data=True, redownload=redownload))
        self.full_res_data = data4d[:,:,:,:-1]
        self.transparency_data = data4d[:,:,:,-1]

    def plot_preview(self, within_range=None, filename=None):
        """
        Plots all previews if within_range is None, or only previews in a given range.
        """
        within_range = CommonUtil.get_within_range(within_range, self.previews.shape[0])
        self._plot_image(self.previews[within_range[0]: within_range[1]] / 255., factor=1, filename=filename)

    def plot_cloud_masks(self, within_range=None, filename=None):
        """
        Plots all cloud masks if within_range is None, or only masks in a given range.
        """
        within_range = CommonUtil.get_within_range(within_range, self.cloud_masks.shape[0])
        self._plot_image(self.cloud_masks[within_range[0]: within_range[1]],
                         factor=1, cmap=plt.cm.binary, filename=filename)

    def _plot_image(self, data, factor=2.5, cmap=None, filename=None):
        rows = data.shape[0] // 5 + (1 if data.shape[0] % 5 else 0)
        aspect_ratio = (1.0 * data.shape[1]) / data.shape[2]
        fig, axs = plt.subplots(nrows=rows, ncols=5, figsize=(15, 3 * rows * aspect_ratio))
        for index, ax in enumerate(axs.flatten()):
            if index < data.shape[0] and index < len(self.dates):
                caption = str(index) + ': ' + self.dates[index].strftime('%Y-%m-%d')
                if self.cloud_coverage is not None:
                    caption = caption + '(' + "{0:2.0f}".format(self.cloud_coverage[index] * 100.0) + '%)'

                ax.set_axis_off()
                ax.imshow(data[index] * factor if data[index].shape[-1] == 3 or data[index].shape[-1] == 4 else
                          data[index] * factor, cmap=cmap, vmin=0.0, vmax=1.0)
                ax.text(0, -2, caption, fontsize=12, color='r' if self.mask[index] else 'g')
            else:
                ax.set_axis_off()

        if filename:
            plt.savefig(self.project_name + '/' + filename, bbox_inches='tight')

    def _load_cloud_masks(self):
        """
        Loads masks from disk, if they already exist.
        """
        cloud_masks_filename = self.project_name + '/cloudmasks/cloudmasks.npy'

        if not os.path.isfile(cloud_masks_filename):
            return False

        with open(cloud_masks_filename, 'rb') as fp:
            self.cloud_masks = np.load(fp)
        return True

    def _save_cloud_masks(self):
        """
        Saves masks to disk.
        """
        cloud_masks_filename = self.project_name + '/cloudmasks/cloudmasks.npy'

        if not os.path.exists(self.project_name + '/cloudmasks'):
            os.makedirs(self.project_name + '/cloudmasks')

        with open(cloud_masks_filename, 'wb') as fp:
            np.save(fp, self.cloud_masks)

    def _run_cloud_detection(self, rerun, threshold):
        """
        Determines cloud masks for each acquisition.
        """
        loaded = self._load_cloud_masks()
        if loaded and not rerun:
            LOGGER.info('Nothing to do. Masks are loaded.')
        else:
            LOGGER.info('Downloading cloud data and running cloud detection. This may take a while.')
            self.cloud_masks = self.cloud_mask_request.get_cloud_masks(threshold=threshold)
            self._save_cloud_masks()

    def mask_cloudy_images(self, rerun=False, max_cloud_coverage=0.1, threshold=None):
        """
        Marks images whose cloud coverage exceeds ``max_cloud_coverage``. Those
        won't be used in timelapse.

        :param rerun: Whether to rerun cloud detector
        :type rerun: bool
        :param max_cloud_coverage: Limit on the cloud coverage of images forming timelapse, 0 <= maxcc <= 1.
        :type max_cloud_coverage: float
        :param threshold:  A float from [0,1] specifying cloud threshold
        :type threshold: float or None
        """
        self._run_cloud_detection(rerun, threshold)

        self.cloud_coverage = np.asarray([self._get_coverage(mask) for mask in self.cloud_masks])

        for index in range(0, len(self.mask)):
            if self.cloud_coverage[index] > max_cloud_coverage:
                self.mask[index] = 1



    def mask_invalid_images(self, max_invalid_coverage=0.1):
        """
        Marks images whose invalid area coverage exceeds ``max_invalid_coverage``. Those
        won't be used in timelapse.

        :param max_invalid_coverage: Limit on the invalid area coverage of images forming timelapse, 0 <= maxic <= 1.
        :type max_invalid_coverage: float
        """

        # low-res and hi-res images/cloud masks may differ, just to be safe
        coverage_fullres = np.asarray([1.0-self._get_coverage(mask) for mask in self.transparency_data])
        coverage_preview = np.asarray([1.0-self._get_coverage(mask) for mask in self.preview_transparency_data])

        self.invalid_coverage = np.array([max(x,y) for x,y in zip(coverage_fullres, coverage_preview)])
        
        for index in range(0, len(self.mask)):
            if self.invalid_coverage[index] > max_invalid_coverage:
                self.mask[index] = 1

    def mask_images(self, idx):
        """
        Mannualy mask images with given indexes.
        """
        for index in idx:
            self.mask[index] = 1

    def unmask_images(self, idx):
        """
        Mannualy unmask images with given indexes.
        """
        for index in idx:
            self.mask[index] = 0

    def create_date_stamps(self):
        """
        Create date stamps to be included to gif.
        """
        filtered = list(compress(self.dates, list(np.logical_not(self.mask))))

        if not os.path.exists(self.project_name + '/datestamps'):
            os.makedirs(self.project_name + '/datestamps')

        for date in filtered:
            TimestampUtil.create_date_stamp(date, filtered[0], filtered[-1],
                                            self.project_name + '/datestamps/' + date.strftime(
                                                "%Y-%m-%dT%H-%M-%S") + '.png')

    def create_timelapse(self, scale_factor=0.3):
        """
        Adds date stamps to full res images and stores them in timelapse subdirectory.
        """
        filtered = list(compress(self.dates, list(np.logical_not(self.mask))))

        if not os.path.exists(self.project_name + '/timelapse'):
            os.makedirs(self.project_name + '/timelapse')

        self.timelapse = [TimestampUtil.add_date_stamp(self._get_filename('fullres', date.strftime("%Y-%m-%dT%H-%M-%S")),
                                         self.project_name + '/timelapse/' + date.strftime(
                                             "%Y-%m-%dT%H-%M-%S") + '.png',
                                         self._get_filename('datestamps', date.strftime("%Y-%m-%dT%H-%M-%S")),
                                         scale_factor=scale_factor) for date in filtered]

    @staticmethod
    def _get_coverage(mask):
        coverage_pixels = np.count_nonzero(mask)
        return 1.0 * coverage_pixels / mask.size

    @staticmethod
    def _iso_to_datetime(date):
        """ Convert ISO 8601 time format to datetime format

        This function converts a date in ISO format, e.g. 2017-09-14 to a datetime instance, e.g.
        datetime.datetime(2017,9,14,0,0)

        :param date: date in ISO 8601 format
        :type date: str
        :return: datetime instance
        :rtype: datetime
        """
        chunks = list(map(int, date.split('T')[0].split('-')))
        return datetime(chunks[0], chunks[1], chunks[2])

    @staticmethod
    def _datetime_to_iso(date, only_date=True):
        """ Convert datetime format to ISO 8601 time format

        This function converts a date in datetime instance, e.g. datetime.datetime(2017,9,14,0,0) to ISO format,
        e.g. 2017-09-14

        :param date: datetime instance to convert
        :type date: datetime
        :param only_date: whether to return date only or also time information. Default is `True`
        :type only_date: bool
        :return: date in ISO 8601 format
        :rtype: str
        """
        if only_date:
            return date.isoformat().split('T')[0]
        return date.isoformat()

    @staticmethod
    def _diff_month(start_dt, end_dt):
        return (end_dt.year - start_dt.year) * 12 + end_dt.month - start_dt.month + 1

    @staticmethod
    def _get_month_list(start_dt, end_dt):
        month_names = {1: 'J', 2: 'F', 3: 'M', 4: 'A', 5: 'M', 6: 'J', 7: 'J', 8: 'A', 9: 'S', 10: 'O', 11: 'N',
                       12: 'D'}

        total_months = SentinelHubTimelapse._diff_month(start_dt, end_dt)
        all_months = list(rrule(MONTHLY, count=total_months, dtstart=start_dt))
        return [month_names[date.month] for date in all_months]

    def _get_filename(self, subdir, date):
        for filename in glob.glob(self.project_name + '/' + subdir + '/*'):
            if date in filename:
                return filename

        return None

    def _get_timelapse_images(self):
        if self.timelapse is None:
            data = np.array(self.fullres_request.get_data())[:,:,:,:-1]
            return [data[idx] for idx, _ in enumerate(data) if self.mask[idx] == 0]
        return self.timelapse

    def make_video(self, filename='timelapse.avi', fps=2, is_color=True, n_repeat=1):
        """
        Creates and saves an AVI video from timelapse into ``timelapse.avi``
        :param fps: frames per second
        :type param: int
        :param is_color:
        :type is_color: bool
        """

        images = np.array([image[:,:,[2,1,0]] for image in self._get_timelapse_images()])

        if None in self.full_size:
            self.full_size = (int(images.shape[2]),int(images.shape[1]))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(os.path.join(self.project_name, filename), fourcc, float(fps), self.full_size,
                                is_color)
                
        for _ in range(n_repeat):
            for image in images:
                video.write(image)
        video.write(images[-1])

        video.release()
        cv2.destroyAllWindows()

    def make_gif(self, filename='timelapse.gif', fps=3):
        """
        Creates and saves a GIF animation from timelapse into ``timelapse.gif``
        :param fps: frames per second
        :type fps: int
        """
        with imageio.get_writer(os.path.join(self.project_name, filename), mode='I', fps=fps) as writer:
            for image in self._get_timelapse_images():
                writer.append_data(image)


class TimestampUtil:
    """
    Utility methods related to timestamps.
    """

    @staticmethod
    def add_date_stamp(input_image_path, output_image_path, watermark_image_path,
                       scale_factor=0.3):

        base_image = Image.open(input_image_path)
        watermark = Image.open(watermark_image_path)

        width, height = base_image.size
        w_width, w_height = watermark.size

        scale = scale_factor * width / w_width

        watermark = watermark.resize((int(scale * w_width), int(scale * w_height)), Image.ANTIALIAS)

        transparent = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        transparent.paste(base_image, (0, 0))
        transparent.paste(watermark, (width - int(scale * w_width), 0), mask=watermark)
        transparent.save(output_image_path)
        # Convert RGBA to RGB and return as numpy
        return np.array(transparent.convert('RGB').getdata()).reshape(height, width, 3).astype(np.uint8)

    @staticmethod
    def create_date_stamp(current_dt, start_dt, end_dt, filename):
        years = TimestampUtil._get_years_in_range(start_dt, end_dt)
        equal_year_size = [1] * len(years)

        months_size = [1] * 12

        # Create colors
        sh_colors = {'light': (173. / 255, 183. / 255, 2. / 255), 'dark': (116. / 255, 110. / 255, 1. / 255)}

        year_colors = [sh_colors['light'] if year <= current_dt.year else sh_colors['dark'] for year in years]
        month_colors = [sh_colors['light'] if index <= current_dt.month else sh_colors['dark'] for index in
                        range(1, 13)]

        # First Ring (outside)
        fig, ax = plt.subplots(figsize=(12.5, 5))
        ax.axis('equal')
        my_pie, texts = ax.pie(equal_year_size, radius=1.5, colors=year_colors,
                               labeldistance=1.05, counterclock=False, startangle=90,
                               textprops={'color': sh_colors['light'], 'weight': 'medium'})

        plt.setp(my_pie, width=0.3, edgecolor=None)

        # Second Ring (Inside)
        my_pie2, texts = ax.pie(months_size, radius=1.5 - 0.3, colors=month_colors,
                                labeldistance=0.9, counterclock=False, startangle=90)

        plt.setp(my_pie2, width=0.4, edgecolor=None)

        if current_dt.day > 9:
            ax.text(-0.6, -0.3, str(current_dt.day), fontsize=100, color=sh_colors['light'], weight='medium')
        else:
            ax.text(-0.3, -0.3, str(current_dt.day), fontsize=100, color=sh_colors['light'], weight='medium')

        ax.text(1.3, 0.8, str(current_dt.year), fontsize=100, color=sh_colors['light'], weight='medium')

        fig.savefig(filename, transparent=True, dpi=300, )
        plt.close()

    @staticmethod
    def _get_years_in_range(start_dt, end_dt):
        return list(range(start_dt.year, end_dt.year + 1))


class CommonUtil:
    @staticmethod
    def get_within_range(within_range, n_imgs):
        """
        Returns the range of images to be plotted.

        :param within_range: tuple of the first and the last image to be plotted, or None
        :type within_range: tuple of ints or None
        :param n_imgs: total number of images
        :type n_imgs: int
        :return: tuple of the first and the last image to be plotted
        :rtype: tuple of two ints
        """
        if within_range is None:
            return [0, n_imgs]
        return max(within_range[0], 0), min(within_range[1], n_imgs)
