import os
import json

from typing import List
from collections import namedtuple
from cv2 import imread

_JSON_FILENAME = "labels.json"

_TRAIN_DIRECTORY_NAME = "train"
_TEST_DIRECTORY_NAME = "test"

_IMGS_SUPPORTED_EXTENSIONS = [".jpg", ".png"]


DatasetImage = namedtuple("DatasetImage", ["file_name", "img", "label"])


class DatasetSplitter(object):
    def __init__(self, dataset_dir):
        """
        :param dataset_dir: Path to the directory of dataset.

        Needed structure of directory:
        dataset_dir/
        |
        |-- train/
        |-- test/

        "train" and "test" directories consist the images and `labels.json` with `yes` and `no` labels for every image in directory.

        Needed structure of `labels.json`:
        {
            "yes": [
                "1.jpg",
                "2.jpg",
                ...
            ],
            "no": [
                "3.jpg",
                "4.jpg",
                ...
            ]
        }
        """
        self._dataset_dir = dataset_dir

        self._train_dir = os.path.normpath(os.path.join(self._dataset_dir, _TRAIN_DIRECTORY_NAME))
        self._test_dir = os.path.normpath(os.path.join(self._dataset_dir, _TEST_DIRECTORY_NAME))

        self._train_imgs: List[DatasetImage] = []
        self._test_imgs: List[DatasetImage] = []

    def _get_info_from_dataset_dir(self, dir):
        """
        :return: tuple of two lists: (images, labels)
        """
        with open(os.path.join(dir, _JSON_FILENAME)) as f:
            data = json.load(f)

        res_imgs = []
        files_to_process = []
        for file in os.listdir(dir):
            name, ext = os.path.splitext(file)
            if ext not in _IMGS_SUPPORTED_EXTENSIONS:
                continue

            files_to_process.append(file)

        files_to_process.sort()
        for file in files_to_process:
            if file in data["yes"]:
                label = "yes"
            else:
                label = "no"

            res_imgs.append(DatasetImage(file_name=os.path.normpath(os.path.join(os.path.basename(dir), file)),
                                         img=imread(os.path.join(dir, file)),
                                         label=label))

        return res_imgs

    def get_train_dataset(self):
        if not self._train_imgs:
            self._train_imgs = self._get_info_from_dataset_dir(self._train_dir)
        return self._train_imgs

    def get_test_dataset(self):
        if not self._test_imgs:
            self._test_imgs = self._get_info_from_dataset_dir(self._test_dir)
        return self._test_imgs
