import os
import logging
import shutil
import numpy as np

from typing import List

from common import DatasetImage
from algorithm.chair import ChairWidthCalculator
from algorithm.door import DoorWidthCalculator


_NUM_ORBS_FEATURES = 8000
_ORB_MATCH_DISTANCE = 70
_VERTICAL_LINE_HOUGH_COEFF = 0.3
_CHAIR_HEIGHT_COEFF = 0.5

_TRAIN_RES_OUTPUT_DIR = "./TRAIN_VERBOSE_OUTPUT"
_TEST_RES_OUTPUT_DIR = "./TEST_VERBOSE_OUTPUT"

_train_logger = logging.getLogger("train_logger")
_test_logger = logging.getLogger("test_logger")


class SemanticGapProblemSolver(object):
    def __init__(self, train_imgs: List[DatasetImage], test_imgs: List[DatasetImage], templates_imgs: List[np.ndarray]):
        self._train_imgs = train_imgs
        self._test_imgs = test_imgs
        self._templates_imgs = templates_imgs

        self._hyperparameters = {
            "num_orbs_features": _NUM_ORBS_FEATURES,
            "orb_match_dist": _ORB_MATCH_DISTANCE,
            "vertical_line_hough_coeff": _VERTICAL_LINE_HOUGH_COEFF,
            "chair_height_coeff": _CHAIR_HEIGHT_COEFF,
        }

        self._train_output_dir = _TRAIN_RES_OUTPUT_DIR
        if os.path.exists(self._train_output_dir):
            shutil.rmtree(self._train_output_dir)
        os.makedirs(self._train_output_dir)

        self._test_output_dir = _TEST_RES_OUTPUT_DIR
        if os.path.exists(self._test_output_dir):
            shutil.rmtree(self._test_output_dir)
        os.makedirs(self._test_output_dir)

    def _find_chair_width(self, img, img_name, logger, output_dir, verbose=True):
        return ChairWidthCalculator.find_chair_width(data_img=img,
                                                     templates_imgs=self._templates_imgs,
                                                     output_dir=output_dir,
                                                     logger=logger,
                                                     hyperparameters=self._hyperparameters,
                                                     img_name=img_name,
                                                     verbose=verbose)

    def _find_door_width(self, img, img_name, logger, output_dir, verbose=True):
        return DoorWidthCalculator.find_door_width(data_img=img,
                                                   output_dir=output_dir,
                                                   logger=logger,
                                                   hyperparameters=self._hyperparameters,
                                                   img_name=img_name,
                                                   verbose=verbose)

    def _chair_is_valid(self, chair_width, door_width):
        # maybe should use some THRESHOLD ???
        return chair_width <= door_width

    def _run_algorithm_on_dataset(self,
                                  dataset_imgs: List[DatasetImage], logger: logging.Logger,
                                  output_dir, verbose=True):
        logger.info("Current num of ORB features: %d" % self._hyperparameters["num_orbs_features"])
        logger.info("Current good match distance: %f" % self._hyperparameters["orb_match_dist"])
        logger.info("Current coefficient of chair's height: %f" % self._hyperparameters["chair_height_coeff"])
        logger.info("Current vertical coefficient for Hough transform algorithm: %f" % self._hyperparameters["vertical_line_hough_coeff"])

        num_of_well_predicted = 0
        for cur_img in dataset_imgs:
            logger.info("Processing image '%s'..." % cur_img.file_name)
            # find the chair and it's width
            chair_width = self._find_chair_width(img=cur_img.img,
                                                 img_name=os.path.basename(cur_img.file_name),
                                                 logger=logger,
                                                 output_dir=output_dir,
                                                 verbose=verbose)

            # find the door and it's width
            door_width = self._find_door_width(img=cur_img.img,
                                               img_name=os.path.basename(cur_img.file_name),
                                               logger=logger,
                                               output_dir=output_dir,
                                               verbose=verbose)

            # compare the two widths
            predicted_label = "yes" if self._chair_is_valid(chair_width, door_width) else "no"

            # compare the result with `label`
            num_of_well_predicted += int(predicted_label == cur_img.label)
            logger.info("Algorithm is done for image '%s'. Predicted: %s. Real: %s"
                        % (cur_img.file_name, predicted_label, cur_img.label))

        accuracy = num_of_well_predicted / len(dataset_imgs)
        return accuracy

    def try_on_train(self, verbose=True):
        _train_logger.info("==== START ALGORITHM ON TRAIN DATASET ====")
        accuracy = self._run_algorithm_on_dataset(dataset_imgs=self._train_imgs,
                                                  logger=_train_logger,
                                                  output_dir=self._train_output_dir,
                                                  verbose=verbose)
        _train_logger.info("CALCULATED ACCURACY ON TRAIN DATASET: %.4f" % accuracy)
        _train_logger.info("==== END ALGORITHM ON TRAIN DATASET ====")

    # or maybe `validate`, not `try_on_test` ???
    def try_on_test(self, verbose=True):
        _test_logger.info("==== START ALGORITHM ON TEST DATASET ====")
        accuracy = self._run_algorithm_on_dataset(dataset_imgs=self._test_imgs,
                                                  logger=_test_logger,
                                                  output_dir=self._test_output_dir,
                                                  verbose=verbose)
        _test_logger.info("CALCULATED ACCURACY ON TEST DATASET: %.4f" % accuracy)
        _test_logger.info("==== END ALGORITHM ON TEST DATASET ====")
