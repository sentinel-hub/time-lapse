# Sentinel-2 time-lapse

The **SentinelHubTimelapse** class allows automatic creation of
time-lapses using our [sentinelhub]() and [s2cloudless]() python packages.

The class allows creation of GIF and AVI videos, given a bounding box and
a time interval. Sentinel-Hub's cloud detection classifier is used to
filter cloudy images out of the time-lapse.

## Requirements

The package requires the following Python packages: (versions listed are
 the versions that we have used):

* [sentinelhub](https://pypi.python.org/pypi/sentinelhub/1.0.0) version 1.0.0
* [s2cloudless](https://pypi.python.org/pypi/s2cloudless/0.0.1) version 0.0.1
* [opencv-python](https://pypi.python.org/pypi/opencv-python) version 3.4.0.12
* [numpy](https://pypi.python.org/pypi/numpy/) version 1.13.3
* [scipy](https://pypi.python.org/pypi/scipy) version 0.19.0
* [matplotlib](https://matplotlib.org) version 2.1.0
* [Pillow](https://pypi.python.org/pypi/Pillow/5.0.0) version 5.0.0
* [imageio](https://pypi.python.org/pypi/imageio/2.2.0) version 2.2.0

## Examples

Examples of how to automatically create time-lapses are provided in
the [SentinelHubTimelapseExamples](./SentinelHubTimelapseExamples.ipynb)
 Jupyter notebook.

## License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a>
<br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
