import os
import time

import logging
# Keep the import below for registering the model definitions
from models import ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import sde_lib
from absl import flags

from torchvision.utils import save_image
from utils import restore_checkpoint

FLAGS = flags.FLAGS


def evaluate(config, workdir, eval_folder, speed_up, freq_mask_path, space_mask_path,alpha):
    sample_dir = os.path.join(workdir, eval_folder)
    os.makedirs(sample_dir, exist_ok=True)
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                            N=int(config.model.num_scales / speed_up))
        sampling_eps = 1e-3 * speed_up
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                               N=int(config.model.num_scales / speed_up))
        sampling_eps = 1e-3 * speed_up
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=int(config.model.num_scales / speed_up))
        sampling_eps = 1e-5 * speed_up
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps,
                                           freq_mask_path, space_mask_path,alpha)
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint.pth')
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())

    logging.info('start sampling!')
    num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1

    for r in range(num_sampling_rounds):
        start = time.time()
        logging.info("sampling -- round: %d" % (r))
        samples, n = sampling_fn(score_model)

        for i in range(samples.shape[0]):
            single_sample = samples[i, ...]
            save_image(single_sample.cpu(),
                       os.path.join(sample_dir, 'image_{}.png'.format(i + r * config.eval.batch_size)))

        logging.info('produce one batch of samples')
        logging.info('one batch cost {}'.format(time.time() - start))
