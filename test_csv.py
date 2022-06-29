import collections
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from timeit import default_timer as timer
from evaluator.evaluator import Evaluator
import datetime
from data_loader.data_loaders import PredictionDataset
from torch.utils.data import DataLoader

# TODO Find solution for PosixPath and WindowsPath
# when model is trained on Linux, it expects a PosixPath to load on Windows as well and vice versa
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


# TODO Assume that ground truth df and pred df are in the exact same label space

def main(config):
    logger = config.get_logger('test')

    evaluator = Evaluator(config)
    start = timer()
    evaluator.evaluate_csv(config["csv_predictor"]["model_preds"], config["csv_predictor"]["ground_truths"])
    end = timer()
    prediction_time = datetime.timedelta(seconds=(end - start))

    evaluator.pred_time = prediction_time

    evaluator.save(type_eval="test")

    log = evaluator.metrics_results
    logger.info(log)



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-p', '--model_preds', default=None, type=str,
                      help="path to csv file with emotion predictions")
    args.add_argument('-t', '--ground_truths', default=None, type=str,
                      help="path to csv file with emotion ground truths")

    config = ConfigParser.from_args(args)
    main(config)
