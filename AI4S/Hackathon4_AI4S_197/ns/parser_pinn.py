import argparse
from datetime import datetime


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--adam_iter', default=10000, type=int
    )
    parser.add_argument(
        '--save_path', default=f'./data/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
    )
    parser.add_argument(
        '--verbose', default=False
    )
    parser.add_argument(
        '--repeat', default=1
    )
    parser.add_argument(
        '--num_layers', default=8
    )
    parser.add_argument(
        '--num_neurons', default=40
    )
    parser.add_argument(
        '--start_epoch', default=0
    )
    parser.add_argument(
        '--print_freq', default=20, type=int
    )
    parser.add_argument(
        '--save_freq', default=1000, type=int
    )
    parser.add_argument(
        '--noise_type', default='t1', type=str
    )
    parser.add_argument(
        '--noise', default=0.1, type=float
    )
    parser.add_argument(
        '--abnormal_ratio', default=0, type=float
    )
    parser.add_argument(
        '--N', default=1000, type=int
    )
    parser.add_argument(
        '--weight', default=1.0, type=float
    )
    return parser
