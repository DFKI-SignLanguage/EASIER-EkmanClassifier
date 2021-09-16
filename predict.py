import collections
from tqdm import tqdm
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from data_loader.data_loaders import PredictionDataset
from torch.utils.data import DataLoader
import pandas as pd
from timeit import default_timer as timer
from evaluator.evaluator import Evaluator
import datetime


# TODO Find solution for PosixPath and WindowsPath
# when model is trained on Linux, it expects a PosixPath to load on Windows as well and vice versa
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


def main(config):
    logger = config.get_logger('inspect')

    # TODO
    """
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=2,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    evaluator = Evaluator(config, data_loader, device)
    evaluator.load_val_eval_df()

    start = timer()
    evaluator.evaluate_model(model)
    end = timer()
    prediction_time = datetime.timedelta(seconds=(end - start))

    evaluator.pred_time = prediction_time

    evaluator.save(type_eval="test")

    log = evaluator.metrics_results
    logger.info(log)
    """
    logger = config.get_logger('test')

    # setup data_loader instances
    pred_dataset = PredictionDataset(config["predictor"]["in_dir"])
    data_loader = DataLoader(pred_dataset, batch_size=1,
                             shuffle=False, num_workers=0)

    pred_df = pd.DataFrame()

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data) in enumerate(tqdm(data_loader)):
            data = data.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Generates the predictions for a given (already trained) model.'
                                               'Predictions are in JSON format as dictionary')
    args.add_argument('-c', '--config', default=None, type=str, required=True,
                      help='config file path')
    args.add_argument('-d', '--device', default=None, type=str, required=False,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-p', "--predict", action="store_true", required=True)

    args.add_argument('-m', '--model', default=None, type=str, required=True,
                      help='path to binary saved prediction model')
    args.add_argument('-i', '--input', default=None, type=str, required=True,
                      help='path to a directory of images to analyse')
    args.add_argument('-o', '--output', default=None, type=str, required=True,
                      help='path to a CSV file that will contain the predictions.'
                           ' CSV header is ["imgname", "neutral", "anger", "disgust", "fear", "happy", "sad", "surprise", "none"].'
                           ' Data is in 1-hot format ')

    config = ConfigParser.from_args(args)

    # config["data_loader"]["args"]["training"] = False
    main(config)
