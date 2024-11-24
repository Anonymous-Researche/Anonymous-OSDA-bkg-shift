import yaml
import hashlib
import time
import importlib
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
import wandb
import numpy as np

def get_dict_hash(dictionary: dict) -> str:
    dhash = hashlib.md5()
    dump = yaml.dump(dictionary)
    encoded = dump.encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def get_class_name(module_class_string, split=None):
    module_name, class_name = module_class_string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    assert hasattr(module, class_name), "class {} is not in {}".format(
        class_name, module_name
    )
    cls = getattr(module, class_name)
    name = cls.name
    if split is not None:
        name += "_" + split
    return name


def load_config(config_file: str) -> dict:
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.Loader)

    return config


def get_random_seed() -> int:
    return int(time.time() * 256) % (2 ** 32)


def filter_config(config: DictConfig) -> dict:
    def is_special_key(key: str) -> bool:
        return key[0] == "_" and key[-1] == "_"

    primitive_config = OmegaConf.to_container(config)

    filt = {
        k: v
        for k, v in primitive_config.items()
        if (not OmegaConf.is_interpolation(config, k))
        # and (not is_special_key(k))
        and v is not None
    }
    return filt


def log_hyperparams(config: DictConfig, trainer: Trainer) -> None:
    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    for key in ["models", "datamodule"]:
        hparams[key] = filter_config(config[key])

    for key in ["seed", "dataset", "arch"]:
        hparams[key] = config[key]

    print(hparams)
    trainer.logger.log_hyperparams(hparams)


def add_to_odict(odict, item):
    if item not in odict:
        ind = len(odict)
        odict[item] = ind

api = wandb.Api()
def get_group_runs(method, seed):
    runs = api.runs("shiftpu", {
        "$and": [{
        'group': {
                "$eq": f'{method}_cifar10_seed_{seed}'
            }
        }]
    })
    return runs

def get_auc_and_fpr(fprs, recalls, aucs, beta, aps=None):
    feasible_records = np.where(fprs > 1-beta)[0]
    if len(feasible_records)==0:
        if aps is None:
            return 0., 0., 1.
        else:
            return 0., 0., 0., 1.
    selected_feasible = np.argmax(recalls[feasible_records])
    alpha = np.max(recalls[feasible_records])
    auc = aucs[feasible_records[selected_feasible]]
    fpr = fprs[feasible_records[selected_feasible]]
    if aps is None:
        return auc, alpha, fpr
    ap = aps[feasible_records[selected_feasible]]
    return auc, ap, alpha, fpr

def get_stats(run):
    stats_names = ['selected AU-ROC', 'selected recall', 'selected fpr',
                   'selected acc', 'selected alpha:']
    results = {stat.split(' ')[1]: run.history()[f'pred/performance.{stat}'].dropna().iloc[-1] for stat in stats_names}
    results['true_alpha'] = run.history()['pred/MPE_estimate_ood.true'].dropna().iloc[-1]
    return results

def select_winning_run():
    seeds = [8, 103, 1057]
    # ground_truth_alphas = [0.01062, 0.01809, 0.06664, 0.04588] + [0.03851, 0.1527, 0.2367, 0.05168] #, 0.005638
    # all_selected_stats = []
    betas_range = np.arange(0.001, 0.02, 0.001)
    # betas_range = np.arange(0.001, 0.05, 0.002)
    aucs = np.zeros((len(seeds), len(betas_range)))
    aps = np.zeros((len(seeds), len(betas_range)))
    alphas = np.zeros((len(seeds), len(betas_range)))
    fprs = np.zeros((len(seeds), len(betas_range)))

    selected_runs = np.zeros((len(seeds), len(betas_range)))
    for i, seed in enumerate(seeds):
        runs = get_group_runs('precision_at_recall', seed)
        selected_stats = None
        selected_alpha = 0.
        for k, run in enumerate(runs):
            has_aps = True
            history = run.scan_history(keys=['pred/performance.val loss source biased',
                                            'pred/performance.recall target',
                                            'pred/performance.curr AU-ROC',
                                            'pred/performance.curr ave-precision'])
            run_fprs = np.array([row['pred/performance.val loss source biased'] for row in history])
            run_recalls = np.array([row['pred/performance.recall target'] for row in history])
            run_aucs = np.array([row['pred/performance.curr AU-ROC'] for row in history])

            run_aps = np.array([row['pred/performance.curr ave-precision'] for row in history])
            if len(run_aps)==0:
                has_aps = False

            for j, beta in enumerate(betas_range):
                if has_aps:
                    cur_auc, cur_ap, cur_alpha, cur_fpr = get_auc_and_fpr(run_fprs, run_recalls, run_aucs, beta, aps=run_aps)
                else:
                    cur_auc, cur_alpha, cur_fpr = get_auc_and_fpr(run_fprs, run_recalls, run_aucs, beta)
                if cur_alpha >= alphas[i, j]:
                    alphas[i, j] = cur_alpha
                    aucs[i, j] = cur_auc
                    if has_aps:
                        aps[i, j] = cur_ap
                    fprs[i, j] = cur_fpr
                    selected_runs[i, j] = k
        print('winning run for seed {}: {}'.format(seed, runs[int(selected_runs[i, j])].name))