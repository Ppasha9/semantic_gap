"""
Semantic Gap problem solver.

Usage:
    main.py (--train | --test) [--dataset-dir=PATH] [--templates-dir=PATH] [--verbose]
    main.py --help

Options:
    --train                 Flag, means that the solver is running in training mode.
    --test                  Flag, means that the solver is running in testing mode.
    --dataset-dir=PATH      Full path to the directory with dataset (default: "./DATA_PHOTOS")
    --templates-dir=PATH    Full path to the directory with templates images with chair (default: "./TEMPLATES")
    --verbose               Verbose mode flag
    -h, --help              Show this message
"""

import os
import docopt

from common import DatasetSplitter, TemplatesReader
from algorithm import SemanticGapProblemSolver
from logger import init_loggers


def run(opts):
    init_loggers()

    dataset_dir = opts["--dataset-dir"] if opts["--dataset-dir"] else "./DATA_PHOTOS"
    if not os.path.exists(dataset_dir):
        print("ERROR: The directory '%s' doesn't exist." % dataset_dir)
        return 1

    print("Loading the dataset from directory '%s'." % dataset_dir)
    dataset_splitter = DatasetSplitter(dataset_dir)
    train_imgs = dataset_splitter.get_train_dataset()
    test_imgs = dataset_splitter.get_test_dataset()

    templates_dir = opts["--templates-dir"] if opts["--templates-dir"] else "./TEMPLATES"
    if not os.path.exists(templates_dir):
        print("ERROR: The directory '%s' doesn't exist." % templates_dir)
        return 1

    print("Loading the templates' images from directory '%s'." % templates_dir)
    templates_reader = TemplatesReader(templates_dir)
    templates_imgs = templates_reader.get_templates()

    problem_solver = SemanticGapProblemSolver(train_imgs, test_imgs, templates_imgs)

    if opts["--train"]:
        problem_solver.try_on_train(verbose=opts["--verbose"])
    elif opts["--test"]:
        problem_solver.try_on_test(verbose=opts["--verbose"])


if __name__ == "__main__":
    run(docopt.docopt(__doc__))
