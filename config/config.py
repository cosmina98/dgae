import os

import yaml
from easydict import EasyDict as edict
import wandb

MY_WB_NAME = 'yobo'

def get_config(args):

    config_dir = f'./config/{args.dataset}_autoencoder.yaml'
    config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
    config.training.betas = (config.training.beta1, config.training.beta2)
    config.dataset = args.dataset
    config.work_type = args.work_type
    config.train_prior = False
    return config


def get_prior_config(args):
    wandb.login(key='c6350accf3d3ceacf6585d4d9515f3ff37db8712')
    api = wandb.Api()
    run_id = args.model_folder[-8:]
    run_name = f'{MY_WB_NAME}/VQ-GAE_{args.dataset}_train_autoencoder/{run_id}'
    run = api.run(run_name)
    config_autoencoder = run.config
    config_dir = f'./config/{args.dataset}_prior.yaml'
    config = yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader)
    config = edict({**config_autoencoder, **config})
    config.training.betas = (config.training.beta1, config.training.beta2)
    config.dataset = args.dataset
    config.work_type = args.work_type
    config.autoencoder_path = args.model_folder
    config.train_prior = True

    #config.data.add_spectral_feat=False
    #config.data.add_cycles_feat=False
    #config.data.add_path_feat=True
    #config.model.quantizer.nc=2
    config.model.quantizer.init_steps = 0
    return config

def get_sample_config(args):
    run_id = args.model_folder[-8:]
    api = wandb.Api()
    run_name = f'{MY_WB_NAME}/VQ-GAE_{args.dataset}_train_prior/{run_id}'
    run = api.run(run_name)
    config = run.config
    config = edict({**config})
    config.folder_name = args.model_folder
    config.betas = (config.training.beta1, config.training.beta2)
    config.dataset = args.dataset
    config.work_type = args.work_type
    config.sample = True
    config.log.wandb = False
    if args.dataset == 'qm9' or args.dataset == 'zinc':
        config.n_samples = 10000
    elif args.dataset == 'enzymes':
        config.n_samples = 117
    elif args.dataset == 'ego':
        config.n_samples = 40
    elif args.dataset == 'community':
        config.n_samples = 20
    else:
        raise NotImplementedError('Dataset not implemented. Check the spelling')
    return config

def only_numerics(seq):
    seq_type = type(seq)
    return seq_type().join(filter(seq_type.isdigit, seq))

