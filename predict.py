import numpy as np
from tqdm import tqdm
import argparse
import torch
import model.model as module_arch
from parse_config import ConfigParser
from data_loader.data_loaders import PredictionDataset
from torch.utils.data import DataLoader
import pandas as pd
import data_loader.data_loaders as module_data


# TODO Find solution for PosixPath and WindowsPath
# when model is trained on Linux, it expects a PosixPath to load on Windows as well and vice versa
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# TODO: Enable this script to load asavchenko models as well
def main(config):
    # logger = config.get_logger('predict')

    # setup data_loader instances
    pred_dataset = PredictionDataset(config["predictor"]["in_dir"], getattr(module_data, config['data_loader']['type']))
    data_loader = DataLoader(pred_dataset, batch_size=1,
                             shuffle=False, num_workers=0)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    # logger.info(model)

    # get function handles of loss and metrics

    # logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location='cpu')
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    out_all_softmax = []
    predictions = []
    img_names = []
    with torch.no_grad():
        for i, (data, img_name) in enumerate(tqdm(data_loader)):
            data = data.to(device)
            output_softmax = model(data)
            output_softmax = output_softmax.cpu().numpy()
            out_all_softmax.append(output_softmax)
            output = np.argmax(output_softmax, axis=1)
            predictions.append(output)

            if type(img_name) == tuple:
                img_name = img_name[0]
            img_names.append(img_name)

    out_all_softmax = np.stack(out_all_softmax).squeeze()

    predictions = np.array(predictions).ravel()
    idx_to_class = data_loader.dataset.idx_to_class
    # out_all_one_hot = np.eye(len(idx_to_class))[predictions]

    pred_class_names = []
    for idx in predictions:
        pred_class_names.append(idx_to_class[idx])

    df_data = {}
    df_data.update({
        "ImageName": img_names})

    for k, v in idx_to_class.items():
        # df_data[v] = out_all_one_hot[:, k]
        df_data[v] = out_all_softmax[:, k]

    df_data.update({
        "ClassName": pred_class_names,
        "Class": predictions
    })

    pred_df = pd.DataFrame(data=df_data)
    try:
        pred_df.to_csv(config["predictor"]["out_dir"])
    except KeyError:
        print(pred_df)
    print(config.resume)
    print(pred_df.ClassName.value_counts())


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Generates the predictions for a given (already trained) model.'
                                               'Predictions are in JSON format as dictionary')
    #args.add_argument('-c', '--config', default=None, type=str, required=True,
    #                  help='config file path')
    args.add_argument('-d', '--device', default=None, type=str, required=False,
                      help='indices of GPUs to enable (default: all)')

    args.add_argument('-p', "--predict", action="store_true", default=True, required=False)

    args.add_argument('-m', '--modeldir', default=None, type=str, required=True,
                      help='path to the directory containing the model and its configuration')
    args.add_argument('-i', '--input', default=None, type=str, required=True,
                      help='path to a directory of images to analyse')
    args.add_argument('-o', '--output', default=None, type=str, required=True,
                      help='path to a CSV file that will contain the predictions.'
                           ' CSV header is ["imgname", "expr1", "expr2", ..., "exprN", "ClassName", "Class"].'
                           ' Expr data is in raw logit values (no softmax applied).')

    config = ConfigParser.from_args(args)

    main(config)
