import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import paddle
import paddle.nn as nn
from pyDOE import lhs
from basic_model import DeepModelSingle
from parser_pinn import get_parser
from gen_poisson_data import get_noise_data, get_truth
from logger import logger

parser_PINN = get_parser()
args = parser_PINN.parse_args()
path = pathlib.Path(args.save_path)
path.mkdir(exist_ok=True, parents=True)
for key, val in vars(args).items():
    print(f"{key} = {val}")
with open(path.joinpath('config'), 'wt') as f:
    f.writelines([f"{key} = {val}\n" for key, val in vars(args).items()])
adam_iter: int = int(args.adam_iter)
bfgs_iter: int = int(args.bfgs_iter)
verbose: bool = bool(args.verbose)
repeat: int = int(args.repeat)
start_epoch: int = int(args.start_epoch)
Nf: int = int(args.Nf)
num_neurons: int = int(args.num_neurons)
num_layers: int = int(args.num_layers)


def gradients(y, x):
    return paddle.grad(y, x, grad_outputs=paddle.ones_like(y),
                       create_graph=True, retain_graph=True, only_inputs=True)[0]

class PINN(DeepModelSingle):
    def __init__(self, planes):
        super(PINN, self).__init__(planes, active=nn.Tanh())

    def equation(self, inn_var, out_var):

        dudx = gradients(out_var, inn_var)
        d2udx2 = gradients(dudx, inn_var)
        eqs = d2udx2 + 16 * paddle.sin(4 * inn_var)
        return eqs

def inference(inn_var, model):

    out_pred = model(inn_var)
    equation = model.equation(inn_var, out_pred)

    return out_pred, equation

def train(X_u_train, X_f_train, u_train, model, Loss_data, Loss_PDE, weight, Optimizer, log_loss):

    Optimizer.clear_grad()
    u_pre = model(X_f_train)
    res_i = model.equation(X_f_train, u_pre)
    u_data_pre = model(X_u_train)
    data_loss = Loss_data(u_train, u_data_pre)
    eqs_loss = Loss_PDE(res_i, paddle.zeros_like(res_i))

    loss_batch = data_loss + weight * eqs_loss
    loss_batch.backward()

    log_loss.append([data_loss.item(), eqs_loss.item()])

    Optimizer.step()

def bfgs_f(params_flatten, X_u_train, X_f_train, u_train, model, log_shape, Loss_data, Loss_PDE, weight):

    params = model.state_dict()
    for i, name in enumerate(params.keys()):
        if i == 0:
            params_flatten_i = params_flatten[0:log_shape[i][0]]
        else:
            params_flatten_i = params_flatten[log_shape[i-1][0]:log_shape[i][0]]
        if len(log_shape[i][1]) == 1:
            params[name] = params_flatten_i
        else:
            params[name] = params_flatten_i.reshape(log_shape[i][1])

    model.set_state_dict(params)
    u_pre = model(X_f_train)
    res_i = model.equation(X_f_train, u_pre)
    u_data_pre = model(X_u_train)
    data_loss = Loss_data(u_train, u_data_pre)
    eqs_loss = Loss_PDE(res_i, paddle.zeros_like(res_i))

    loss_batch = data_loss + weight * eqs_loss
    return loss_batch

def predict_error(model):
    x, u = get_truth(10000, full=True)
    x = paddle.to_tensor(x, dtype='float32', stop_gradient=False)
    u_pred, equation = inference(x, model)
    error_u = np.linalg.norm(u - u_pred.numpy(), 2) / np.linalg.norm(u, 2)
    return error_u
#
def plot_result(model, X_u_train, u_train, filename):
    X_full, u_full = get_truth(N=1000, full=True)
    X_full = paddle.to_tensor(X_full, dtype='float32', stop_gradient=False)
    u_pred, equation = inference(X_full, model)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=200)
    fig.set_tight_layout(True)
    ax.plot(X_full.numpy().ravel(), u_pred.numpy().ravel())
    ax.plot(X_full.numpy().ravel(), u_full.ravel(), '--')
    ax.scatter(X_u_train, u_train, s=4)
    ax.set_ylim([-1., 3])
    plt.savefig(filename)
    plt.close()

def run_experiment(epoch_num, noise_type, noise, loss_type, N, weight, _data=[], abnormal_size=0):
    if paddle.is_compiled_with_cuda():
        paddle.device.set_device('gpu:0')
    else:
        paddle.device.set_device('cpu')

    planes = [1] + [num_neurons] * num_layers + [1]
    # Model
    Net_model = PINN(planes=planes)
    # Loss
    if loss_type == "square":
        Loss_data = nn.MSELoss()
    elif loss_type == "l1":
        Loss_data = nn.L1Loss()
    else:
        raise NotImplementedError(f'Loss type {loss_type} not implemented.')
    Loss_PDE = nn.MSELoss()
    # 下降策略
    # Scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.001, milestones=[1000, ], gamma=0.1)
    # 优化算法
    # Optimizer = paddle.optimizer.Adam(learning_rate=Scheduler, parameters=Net_model.parameters(), beta1=0.8, beta2=0.9)
    Optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=Net_model.parameters())
    lb = np.array([-np.pi, ])
    rb = np.array([np.pi, ])
    ###########################

    if len(_data) == 0:
        X_u_train, u_train = get_noise_data(N, noise_type=noise_type, sigma=noise, size=abnormal_size)
        if Nf == 0:
            X_f_train = X_u_train
        else:
            X_f_train = lb + (rb - lb) * lhs(1, Nf)
            print(X_u_train.shape, X_f_train.shape)
            X_f_train = np.concatenate([X_u_train, X_f_train], axis=0)
        _data.append((X_u_train, X_f_train, u_train))
    X_u_train, X_f_train, u_train = _data[0]
    X_u_train = paddle.to_tensor(X_u_train, dtype='float32', stop_gradient=False)
    X_f_train = paddle.to_tensor(X_f_train, dtype='float32', stop_gradient=False)
    u_train = paddle.to_tensor(u_train, dtype='float32', stop_gradient=False)

    log_loss = []
    print_freq = 20
    sta_time = time.time()

    for it in range(adam_iter):

        learning_rate = Optimizer.get_lr()
        train(X_u_train, X_f_train, u_train, Net_model, Loss_data, Loss_PDE, weight, Optimizer, log_loss)
        if (it + 1) % print_freq == 0:
            print('epoch: {:6d}, lr: {:.3e}, data_loss: {:.3e}, pde_loss: {:.3e}, cost: {:.2f}'.
              format(it, learning_rate, log_loss[-1][0], log_loss[-1][-1], time.time() - sta_time))

    Optimizer._learning_rate = 1e-20
    for it in range(100):
        train(X_u_train, X_f_train, u_train, Net_model, Loss_data, Loss_PDE, weight, Optimizer, log_loss)
    paddle.save({'epoch': adam_iter, 'log_loss': log_loss,
                 'model': Net_model.state_dict(), "optimizer": Optimizer.state_dict()},
        os.path.join(path, f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}.pdparams'))
    error_u = predict_error(Net_model)
    plot_result(Net_model, X_u_train, u_train,
                path.joinpath(f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}.png'))
    with open(path.joinpath('result.csv'), 'a+') as f:
        f.write(f"{epoch_num},{loss_type},{N},{noise_type},{noise},{abnormal_size},{weight},{error_u}\n")
    logger.info(f"Eu: {error_u * 100:.3f}%")

def get_last_idx(filename):
    if not os.path.exists(filename):
        return -1
    with open(filename, "r") as f1:
        last_idx = int(f1.readlines()[-1].strip().split(',')[0])
        return last_idx

# if __name__ == "__main__":
#     N = 1000
#     noise = 0
#     abnormal_ratio = 0.2
#     abnormal_size = int(N * abnormal_ratio)
#     noise_type = 'outlinear'
#     _data = []
#     weight = 1.0
#     run_experiment(0, N=N, noise=noise, noise_type=noise_type, weight=weight,
#                    loss_type='l1', _data=_data, abnormal_size=abnormal_size)