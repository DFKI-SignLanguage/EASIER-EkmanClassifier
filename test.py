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


# TODO Find solution for PosixPath and WindowsPath
# when model is trained on Linux, it expects a PosixPath to load on Windows as well and vice versa
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


def main(config):
    logger = config.get_logger('test')
    # config.mk_save_eval_dir()

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=2,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=0
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

    evaluator = Evaluator(data_loader, device)
    evaluator.set_config(config)
    evaluator.idx_to_class = getattr(module_data, config["test_predictor"]['ground_truths_data_loader']['type']).get_label_map()
    evaluator.load_validation_eval_df()

    model_preds_idx_to_class = getattr(module_data, config["test_predictor"]['model_preds_data_loader']['type']).get_label_map()
    evaluator.convert_idx_to_dataset(model_preds_idx_to_class)

    start = timer()
    evaluator.evaluate_model(model)
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
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    # args.add_argument('-t', '--ground_truths', default=None, type=str,
    #                   help="path to csv file with emotion ground truths")
    args.add_argument('-g', '--ground_truths_data_loader', default=None, type=str,
                      help="flag that specifies the data loader that contains the prediction labels in the style of "
                           "the supplied ground truth csv")
    args.add_argument('-m', '--model_preds_data_loader', default=None, type=str,
                      help="flag that specifies the data loader that contains the prediction labels in the style of "
                           "that the model was trained on")

    config = ConfigParser.from_args(args)
    main(config)
