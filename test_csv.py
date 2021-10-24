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


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    # pred_dataset = PredictionDataset(config["predictor"]["in_dir"], getattr(module_data, config['data_loader']['type']))
    # data_loader = DataLoader(pred_dataset, batch_size=1,
    #                          shuffle=False, num_workers=0)
    evaluator = Evaluator(config)
    evaluator.idx_to_class = getattr(module_data, config["csv_predictor"]['ground_truths_data_loader']['type']).get_label_map()
    evaluator.load_val_eval_df()

    model_preds_idx_to_class = getattr(module_data, config["csv_predictor"]['model_preds_data_loader']['type']).get_label_map()
    evaluator.convert_idx_to_dataset(model_preds_idx_to_class)

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
    args.add_argument('-m', '--model_preds_data_loader', default=None, type=str,
                      help="flag that specifies the data loader that contains the prediction labels in the style of "
                           "that the model was trained on")
    args.add_argument('-p', '--model_preds', default=None, type=str,
                      help="path to csv file with emotion predictions")
    args.add_argument('-t', '--ground_truths', default=None, type=str,
                      help="path to csv file with emotion ground truths")
    args.add_argument('-g', '--ground_truths_data_loader', default=None, type=str,
                      help="flag that specifies the data loader that contains the prediction labels in the style of "
                           "the supplied ground truth csv")
    # args = args.parse_args()
    # config = {}
    # csv_predictor = {
    #     "csv_predictor":
    #         {
    #             "model_preds": args.model_preds,
    #             "ground_truths": args.ground_truths
    #         },
    #     "model_preds_data_loader": {
    #         "type": args.model_preds_data_loader
    #     },
    #     "ground_truths_data_loader":
    #         {
    #             "type": args.ground_truths_data_loader
    #         }
    # }
    # config.update(csv_predictor)

    config = ConfigParser.from_args(args)
    main(config)
