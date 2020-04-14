import os

from cv2 import imread


class TemplatesReader(object):
    def __init__(self, templates_dir):
        """
        :param templates_dir: Path to the directory with templates images.
        This directory should consists only images, that are templates for chair detection.

        Needed structure of directory:
        templates_dir/
        |
        |-- image_1.jpg
        |-- image_2.jpg
        |-- ...

        """
        self._templates_dir = templates_dir
        self._templates_imgs = []

    def get_templates(self):
        if not self._templates_imgs:
            for file in os.listdir(self._templates_dir):
                file_path = os.path.join(self._templates_dir, file)
                self._templates_imgs.append(imread(file_path))
        return self._templates_imgs
