import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json


# Standard file names for models stored in standalone directory
MODEL_BIN = "model_best.pth"
CONFIG_FILE = "config.json"


# TODO Move file from top level to another folder
# TODO Idea to solve config parser problems: Maybe create another config parser (for predict and test_csv) that is
#  similar to the one used for training and testing
# TODO Make the flags similar in all the scripts
# TODO Ensure saving models and testing does not create unnecessary folders
# TODO Allow users to input save location either in config or as arg  ==> If nothing is specified use default locations
class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])
        save_eval_dir = Path(self.config['evaluation_store']['args']['save_dir'])

        # TODO: Save info as
        #  saved/exper_name/run_id/testset_pred

        exper_name = self.config['name']
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%Y%m%d_%H%M%S')
        self.run_id = run_id
        self._save_dir = save_dir / "_".join([exper_name, run_id]) / 'models'
        self._log_dir = save_dir / "_".join([exper_name, run_id]) / 'log'

        self._save_eval_dir = save_eval_dir / "_".join([exper_name, run_id]) / "eval"

        # make directory for saving checkpoints and log and evaluation metrics.
        exist_ok = run_id == ''
        # self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if hasattr(args, "device") and args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        cfg_fname = None
        if hasattr(args, "resume") and args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        elif hasattr(args, "modeldir") and args.modeldir is not None:
            # Filenames for self-contained trained model
            model_dir_path = Path(args.modeldir)
            resume = model_dir_path / MODEL_BIN
            cfg_fname = model_dir_path / CONFIG_FILE
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)

        # Load the "default" config
        config = read_json(cfg_fname)

        # If specified, merge the config with the one specified in the command line
        if hasattr(args, "config") and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        if hasattr(args, "predict"):
            predictor = {
                "predictor":
                    {
                        "in_dir": args.input,
                        "out_dir": args.output,
                    }
            }
            config.update(predictor)

        if hasattr(args, "resume") and hasattr(args, "ground_truths_data_loader"):
            test_predictor = {
                "test_predictor":
                    {
                        "model_preds_data_loader": {
                            "type": args.model_preds_data_loader
                        },
                        "ground_truths_data_loader":
                            {
                                "type": args.ground_truths_data_loader
                            }
                    },

            }
            config.update(test_predictor)

        if hasattr(args, "model_preds") and hasattr(args, "ground_truths"):
            csv_predictor = {
                "csv_predictor":
                    {
                        "model_preds": args.model_preds,
                        "ground_truths": args.ground_truths,
                        # "normalized_label_map": args.normalized_label_map
                    },

            }
            config.update(csv_predictor)

        # parse custom cli options into dictionary
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                       self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def save_eval_dir(self):
        return self._save_eval_dir

    @property
    def log_dir(self):
        return self._log_dir

    def mk_save_dir(self):
        exist_ok = self.run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        write_json(self.config, self.save_dir / 'config.json')

    def mk_save_eval_dir(self):
        exist_ok = self.run_id == ''
        self.save_eval_dir.mkdir(parents=True, exist_ok=exist_ok)


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
