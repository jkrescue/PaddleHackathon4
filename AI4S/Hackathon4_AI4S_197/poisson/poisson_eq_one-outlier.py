"""_summary_
while true; do python poisson_eq_size.py --save_path='./data/test_size' --adam_iter=15000; done
"""
from poisson_eq import *

if __name__ == "__main__":
    idx = 0
    last_idx = get_last_idx(path.joinpath('result.csv'))
    executed_flag = False

    for noise_type in ['one-outlinear']:
        for noise in [0]:
            for abnormal_ratio in [0]:
                for N in [500]:
                    _data = []
                    abnormal_size = int(N * abnormal_ratio)
                    for weight in [1E-0]:
                        if idx > last_idx:
                            run_experiment(idx, noise_type, noise, 'square', N, weight, _data,
                                           abnormal_size=abnormal_size)
                            # executed_flag = True
                        idx += 1

                        if idx > last_idx:
                            run_experiment(idx, noise_type, noise, 'l1', N, weight, _data,
                                           abnormal_size=abnormal_size)
                            # executed_flag = True
                        idx += 1

