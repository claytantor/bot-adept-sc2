#!python
"""
Copyright (C) 2018 Heron Systems, Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import os

import torch
from absl import flags
from dotenv import load_dotenv, find_dotenv
from adept.containers import Local
from adept.utils.script_helpers import make_agent, make_network, make_env, get_head_shapes, count_parameters
from adept.utils.logging import make_log_id, make_logger, print_ascii_logo, log_args, write_args_file, ModelSaver
from tensorboardX import SummaryWriter

# hack to use argparse for SC2
FLAGS = flags.FLAGS
FLAGS(['local.py'])


AdamOptimizer_learning_rate = 0.0005
ACAgent_optimizer = "RMSprop"

# tf.train.AdamOptimizer.learning_rate = 0.0005
ACAgent_entropy_coef = 0.0005
ACAgent_clip_grads_norm = 100.0

ACAgent_normalize_returns = False
ACAgent_normalize_advantages = False
ACAgent_momentum = 0.98


# hyper parameters
# ACAgent_EPS_START = 0.99  # e-greedy threshold start value
# ACAgent_EPS_END = 0.05  # e-greedy threshold end value
# ACAgent_EPS_DECAY = 200  # e-greedy threshold decay
# ACAgent_GAMMA = 0.8  # Q-learning discount factor
# # LR = 0.001  # NN optimizer learning rate
# BATCH_SIZE = 64  # Q-learning batch size


# params for generic actor-critic:
# ==========================================
# ACAgent_model_fn = @build_fully_conv
# ACAgent_policy_cls = @SC2MultiPolicy

# ACAgent_optimizer = @tf.train.AdamOptimizer()
# tf.train.AdamOptimizer.learning_rate = 0.0007

ACAgent_eps = 1e-5

ACAgent_weight_decay=0.0005
ACAgent_value_coef = 0.5
ACAgent_entropy_coef = 0.001

ACAgent_batch_sz = 32
ACAgent_traj_len = 16

ACAgent_discount_alpha = 0.99
ACAgent_discount_gamma = 0.99

ACAgent_gae_lambda = 0.0

ACAgent_clip_rewards = 0.0
ACAgent_clip_grads_norm = 10.0

ACAgent_normalize_returns = False
ACAgent_normalize_advantages = False

# params for A2C:
# ==========================================
# ...

# params for PPO:
# ==========================================
PPOAgent_gae_lambda = 0.95

PPOAgent_n_epochs = 3
PPOAgent_minibatch_sz = 128
PPOAgent_clip_ratio = 0.2
PPOAgent_clip_value = 0.0

PPOAgent_normalize_advantages = True

def main(args):
    load_dotenv(find_dotenv())
  
    # construct logging objects
    print_ascii_logo()
    log_id = make_log_id(args.tag, args.mode_name, args.agent, args.vision_network + args.network_body)
    log_id_dir = os.path.join(args.log_dir, args.env_id, log_id)

    os.makedirs(log_id_dir)
    logger = make_logger('Local', os.path.join(log_id_dir, 'train_log.txt'))
    summary_writer = SummaryWriter(log_id_dir)
    saver = ModelSaver(args.nb_top_model, log_id_dir)

    log_args(logger, args)
    write_args_file(log_id_dir, args)

    # construct env
    env = make_env(args, args.seed)

    # construct network
    torch.manual_seed(args.seed)
    network_head_shapes = get_head_shapes(env.action_space, env.engine, args.agent)
    network = make_network(env.observation_space, network_head_shapes, args)
    logger.info('Network Parameter Count: {}'.format(count_parameters(network)))

    # construct agent
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    agent = make_agent(network, device, env.engine, env.gpu_preprocessor, args)

    # Construct the Container
    def make_optimizer(params):
        print("Optimizer:", ACAgent_optimizer)
        if ACAgent_optimizer == 'Adam':
            return torch.optim.Adam(params, lr=args.learning_rate, eps=ACAgent_eps, weight_decay=ACAgent_weight_decay, amsgrad=False)
        elif ACAgent_optimizer == 'Nesterov':
            return torch.optim.SGD(params, lr=args.learning_rate, momentum=ACAgent_momentum, nesterov=True)
        else:
            return torch.optim.RMSprop(params, lr=args.learning_rate, eps=ACAgent_eps, alpha=ACAgent_discount_alpha)

    container = Local(
        agent,
        env,
        make_optimizer,
        args.epoch_len,
        args.nb_env,
        logger,
        summary_writer,
        args.summary_frequency,
        saver
    )

    # Run the container
    if args.profile:
        try:
            from pyinstrument import Profiler
        except:
            raise ImportError('You must install pyinstrument to use profiling.')
        profiler = Profiler()
        profiler.start()
        container.run(10e3)
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
    else:
        container.run(args.max_train_steps)
    env.close()


if __name__ == '__main__':
    import argparse
    from adept.utils.script_helpers import add_base_args, parse_bool

    parser = argparse.ArgumentParser(description='AdeptRL Local Mode')
    parser = add_base_args(parser)
    parser.add_argument('--gpu-id', type=int, default=0, help='Which GPU to use for training (default: 0)')
    parser.add_argument(
        '-vn', '--vision-network', default='FourConv',
        help='name of preset network (default: FourConv)'
    )
    parser.add_argument(
        '-dn', '--discrete-network', default='Identity',
    )
    parser.add_argument(
        '-nb', '--network-body', default='LSTM',
    )
    parser.add_argument(
        '--agent', default='ActorCritic',
        help='name of preset agent (default: ActorCritic)'
    )
    parser.add_argument(
        '--profile', type=parse_bool, nargs='?', const=True, default=False,
        help='displays profiling tree after 10e3 steps (default: False)'
    )
    parser.add_argument(
        '--debug', type=parse_bool, nargs='?', const=True, default=False,
        help='debug mode sends the logs to /tmp/ and overrides number of workers to 3 (default: False)'
    )

    args = parser.parse_args()

    if args.debug:
        args.nb_env = 3
        args.log_dir = '/tmp/'

    args.mode_name = 'Local'
    main(args)
