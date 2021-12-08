import torch

import utils
import data_loader

from trainer import Trainer
from config import get_config

import os
import time


def main(config):
    utils.prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {"num_workers": 1, "pin_memory": True}

    # instantiate data loaders
    if config.is_train:
        dloader = data_loader.get_train_valid_loader(
            config.data_dir,
            config.batch_size,
            config.random_seed,
            config.valid_size,
            config.shuffle,
            config.show_sample,
            **kwargs,
        )
    else:
        dloader = data_loader.get_test_loader(
            config.data_dir, config.batch_size, **kwargs,
        )
        
    if not config.is_train:
        # omg im gonna cry
        print('using hard-coded path')
        config.model_name = config.hard_coded_path
    else:
        config.model_name = "{}ram_{}_{}x{}_{}--hidden_size{}--glimpse_hidden_layer{}_{}".format(
                "stop" if config.include_stop else "",
                config.num_glimpses,
                config.patch_size,
                config.patch_size,
                config.glimpse_scale,
                config.hidden_size,
                config.glimpse_hidden,
                time.strftime("%Y_%d_%b_%H_%M")
            )

    config.model_dir = os.path.join(config.model_dir, config.model_name)
    if not os.path.exists(config.model_dir):
      os.mkdir(config.model_dir)

    config.plot_dir = "./plots/" + config.model_name + "/"
    if not os.path.exists(config.plot_dir):
      os.makedirs(config.plot_dir)

    trainer = Trainer(config, dloader)

    # either train
    if config.is_train:
        utils.save_config(config)
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
