import argparse
from parse_config import ConfigParser
from timeit import default_timer as timer
from evaluator.evaluator import Evaluator
import datetime


# TODO Find solution for PosixPath and WindowsPath
# when model is trained on Linux, it expects a PosixPath to load on Windows as well and vice versa
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


# TODO Assume that ground truth df and pred df are in the exact same label space

def main(config):
    evaluator = Evaluator()
    start = timer()
    evaluator.evaluate_csv(config)
    end = timer()
    prediction_time = datetime.timedelta(seconds=(end - start))

    evaluator.pred_time = prediction_time

    evaluator.save(save_path=config["out_path"], type_eval="test")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Script to evaluate model predictions by comparing with the '
                                               'ground truths. Both predicitons and ground truths are normalized to '
                                               'match the EASIER label format.')
    args.add_argument('-p', '--model_preds', default=None, type=str,
                      help="path to csv file with emotion predictions")
    args.add_argument('-t', '--ground_truths', default=None, type=str,
                      help="path to csv file with emotion ground truths")
    args.add_argument('-o', '--out_csv_file', default=None, type=str,
                      help="path to csv file to save the evaluation csv results")

    args = args.parse_args()

    config = {
        "model_preds": args.model_preds,
        "ground_truths": args.ground_truths,
        "out_path": args.out_csv_file
    }

    main(config)
