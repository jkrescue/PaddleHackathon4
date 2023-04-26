from unsteady_NS import *

if __name__ == "__main__":
    last_idx = get_last_idx(path.joinpath('result.csv'))
    idx = last_idx + 1
    _data = []
    abnormal_size = int(args.N * args.abnormal_ratio)
    if idx > last_idx:
        run_experiment(idx, N=args.N, noise=args.noise, noise_type=args.noise_type, weight=1E-0,
                        loss_type='l1', _data=_data, abnormal_size=abnormal_size)
        # executed_flag = True
    idx += 1
    if idx > last_idx:
        run_experiment(idx, N=args.N, noise=args.noise, noise_type=args.noise_type, weight=1E-0,
                    loss_type='square', _data=_data, abnormal_size=abnormal_size)