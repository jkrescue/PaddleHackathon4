import os
import sys
import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import math
import pathlib
import paddle
import paddle.nn as nn
import paddle.static as static
from pyDOE import lhs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from basic_model import DeepModelSingle, DeepModelMulti
from parser_pinn import get_parser
from gen_ns_data import get_noise_data, get_truth
from scipy.special import erfinv

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;

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
num_neurons: int = int(args.num_neurons)
num_layers: int = int(args.num_layers)


class PINN_NS_unsteady(DeepModelSingle):
    def __init__(self, planes):
        super(PINN_NS_unsteady, self).__init__(planes, active=nn.Tanh())
        self.lambda_1 = paddle.create_parameter(shape=[1, ], dtype='float32',
                                                default_initializer=nn.initializer.Constant(0.0))
        self.add_parameter("lambda_1", self.lambda_1)
        self.lambda_2 = paddle.create_parameter(shape=[1, ], dtype='float32',
                                                default_initializer=nn.initializer.Constant(0.0))
        self.add_parameter("lambda_2", self.lambda_2)

    def gradients(self, y, x):
        return paddle.grad(y, x, grad_outputs=paddle.ones_like(y),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    def get_lambda_1(self):
        return paddle.to_tensor(self.lambda_1, stop_gradient=False)

    def get_lambda_2(self):
        return paddle.to_tensor(self.lambda_2, stop_gradient=False)

    def out_transform(self, inn_var):
        out_var = self.forward(inn_var)
        psi = out_var[..., 0:1]
        p = out_var[..., 1:2]
        dpsida = paddle.incubate.autograd.grad(psi, inn_var)
        u, v = dpsida[:, 1:2], -dpsida[:, 0:1]
        return p, u, v

    def equation(self, inn_var):
        p, u, v = self.out_transform(inn_var)

        duda = paddle.incubate.autograd.grad(u, inn_var)
        dvda = paddle.incubate.autograd.grad(v, inn_var)
        dpda = paddle.incubate.autograd.grad(p, inn_var)

        dudx, dudy, dudt = duda[..., 0:1], duda[..., 1:2], duda[..., 2:3]
        dvdx, dvdy, dvdt = dvda[..., 0:1], dvda[..., 1:2], dvda[..., 2:3]
        dpdx, dpdy, dpdt = dpda[..., 0:1], dpda[..., 1:2], dpda[..., 2:3]

        d2udx2 = paddle.incubate.autograd.grad(dudx, inn_var)[..., 0:1]
        d2udy2 = paddle.incubate.autograd.grad(dudy, inn_var)[..., 1:2]
        d2vdx2 = paddle.incubate.autograd.grad(dvdx, inn_var)[..., 0:1]
        d2vdy2 = paddle.incubate.autograd.grad(dvdy, inn_var)[..., 1:2]

        res_u = dudt + (u * dudx + v * dudy) * self.get_lambda_1() + dpdx - (d2udx2 + d2udy2) * self.get_lambda_2()
        res_v = dvdt + (u * dvdx + v * dvdy) * self.get_lambda_1() + dpdy - (d2vdx2 + d2vdy2) * self.get_lambda_2()

        return res_u, res_v, p, u, v

    def train(self, DATA, Loss_data, Loss_PDE, weight, Optimizer, log_loss):
        inn_data = DATA[:, 0:3]
        out_data = DATA[:, 3:5]
        Optimizer.clear_grad()
        res_u, res_v, p_pre, u_pre, v_pre = self.equation(inn_data)
        u_loss = paddle.norm(u_pre - out_data[:, 3], p=1) / u_pre.shape[0]
        v_loss = paddle.norm(v_pre - out_data[:, 4], p=1) / v_pre.shape[0]
        eqsU_loss = paddle.norm(res_u, p=2) ** 2 / res_u.shape[0]
        eqsV_loss = paddle.norm(res_v, p=2) ** 2 / res_v.shape[0]
        data_loss = u_loss + v_loss
        eqs_loss = eqsU_loss + eqsV_loss

        loss_batch = data_loss + weight * eqs_loss
        loss_batch.backward()

        log_loss.append([data_loss.item(), eqs_loss.item()])

        Optimizer.step()

    def predict_error(self, p_pre, u_pre, v_pre, lambda_1, lambda_2):
        x, y, t, u, v, p = get_truth()
        error_u = np.linalg.norm(u_pre - u, 2) / np.linalg.norm(u, 2)
        error_v = np.linalg.norm(v_pre - v, 2) / np.linalg.norm(v, 2)
        error_vel = np.sqrt(np.sum((u_pre - u) ** 2 + (v_pre - v) ** 2)) / np.sqrt(np.sum(u ** 2 + v ** 2))
        error_max = np.max(np.sqrt((u_pre - u) ** 2 + (v_pre - v) ** 2))
        error_p = np.linalg.norm(p_pre - p, 2) / np.linalg.norm(p, 2)
        error_lambda_1 = (np.abs((lambda_1 - 1.)))
        error_lambda_2 = (np.abs((lambda_2 - 0.01) / 0.01))

        _shape = (50, 100)
        t_p = 100
        p_pred_slice = p_pre[t_p::200, 0]
        p_slice = p[t_p::200, 0]
        p_mean_slice = np.mean(p_pred_slice - p_slice)
        error_p1 = np.linalg.norm(p_pred_slice - p_slice - p_mean_slice, 2) / np.linalg.norm(p_slice, 2)

        return error_u, error_v, error_vel, error_max, error_p, error_p1, error_lambda_1, error_lambda_2

    def plot_result(self, p_pre, u_pre, v_pre, filename):
        x, y, t, u, v, p = get_truth()
        _shape = (50, 100)
        t_p = 100
        x = x[t_p::200, 0]
        y = y[t_p::200, 0]
        t = t[t_p::200, 0]
        u = u[t_p::200, 0]
        v = v[t_p::200, 0]
        p = p[t_p::200, 0]
        p_pre = p_pre[t_p::200, 0]
        u_pre = u_pre[t_p::200, 0]
        v_pre = v_pre[t_p::200, 0]
        p_mean_slice = np.mean(p_pre - p)

        plt.clf()
        fig, ax = plt.subplots(6, 1, figsize=(10, 20), dpi=200)
        fig.set_tight_layout(True)
        im = ax[0].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape), u_pre.ravel().reshape(*_shape),
                              shading='gouraud')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[1].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape), u.ravel().reshape(*_shape),
                              shading='gouraud')
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[2].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape),
                              u_pre.ravel().reshape(*_shape) - u.ravel().reshape(*_shape), cmap='bwr',
                              shading='gouraud')
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[3].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape),
                              (p_pre - p_mean_slice).ravel().reshape(*_shape), shading='gouraud')
        divider = make_axes_locatable(ax[3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[4].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape), p.ravel().reshape(*_shape),
                              shading='gouraud')
        divider = make_axes_locatable(ax[4])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[5].pcolormesh(x.ravel().reshape(*_shape), y.ravel().reshape(*_shape),
                              (p_pre - p_mean_slice).ravel().reshape(*_shape) - p.ravel().reshape(*_shape), cmap='bwr',
                              shading='gouraud')
        divider = make_axes_locatable(ax[5])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax[5].set_title(f'mse: {np.mean(np.abs(p_pre - p)):.4f}')

        plt.colorbar(im, cax=cax)
        plt.savefig(filename)


def sieve_obs(DATA, u_pre, v_pre, ratio):
    u_data = DATA[:, 0:1]
    v_data = DATA[:, 1:2]
    data_erro = np.sqrt(np.square(u_pre - u_data) + np.square(v_pre - v_data))
    error = data_erro.ravel()
    # print("data erro:", error)
    execluded_number = int(len(error) * ratio)
    print(f'{execluded_number} observations have been removed.')
    ind_rm = np.argpartition(error, -execluded_number)[-execluded_number:]

    # x_data = DATA.numpy()[ind_rm, 0:1]
    # print("sieve:", x_data.shape)
    return ind_rm


def sieve_obs_sigma(DATA, u_pre, v_pre, k=2):
    u_data = DATA[:, 0:1]
    v_data = DATA[:, 1:2]
    data_erro = np.sqrt(np.square(u_pre - u_data) + np.square(v_pre - v_data))
    error = data_erro.ravel()
    sigma = np.median(np.abs(error)) / math.sqrt(2) / erfinv(0.5)
    ind_rm = (np.abs(error) > k * sigma)
    # print("data erro:", error)
    original_N = len(error)

    x_data = DATA[ind_rm, 0:1]
    N_rm = len(x_data)
    print(f'{N_rm} observations have been removed.')
    return ind_rm


def plot_loss(x, y, label, title=None, color=None, marker=None):
    # sbn.set_style('ticks')
    # sbn.set(color_codes=True)
    font = {'weight': 'normal', 'size': 30}
    plt.plot(x, y, label=label, color=color, marker=marker)
    plt.semilogy()
    plt.grid(True)  # 添加网格
    plt.legend(loc="upper right", prop=font)
    plt.xlabel('iterations', font)
    plt.ylabel('loss value', font)
    plt.yticks(size=font["size"])
    plt.xticks(size=font["size"])
    plt.title(title, font)
    # plt.pause(0.001)


def build(model, x_train, N_truth, weight, switch):
    Tra_inn = paddle.static.data('Tra_inn', shape=[x_train.shape[0], 3], dtype='float32')
    Tra_inn.stop_gradient = False
    Tra_out = paddle.static.data('Tra_out', shape=[x_train.shape[0], 2], dtype='float32')
    # Tra_out.stop_gradient = False

    Val_inn = paddle.static.data('Val_inn', shape=[N_truth, 3], dtype='float32')
    res_u, res_v, P_tra, U_tra, V_tra = model.equation(Tra_inn)

    u_loss = switch * paddle.norm(U_tra - Tra_out[:, 0:1], p=2) ** 2 / U_tra.shape[0] + \
             (1 - switch) * paddle.norm(U_tra - Tra_out[:, 0:1], p=1) / U_tra.shape[0]
    v_loss = paddle.norm(V_tra - Tra_out[:, 1:2], p=2) ** 2 / V_tra.shape[0] + \
             (1 - switch) * paddle.norm(V_tra - Tra_out[:, 1:2], p=1) / V_tra.shape[0]

    eqsU_loss = paddle.norm(res_u, p=2) ** 2 / res_u.shape[0]
    eqsV_loss = paddle.norm(res_v, p=2) ** 2 / res_v.shape[0]
    data_loss = u_loss + v_loss
    eqs_loss = eqsU_loss + eqsV_loss

    loss_batch = data_loss + weight * eqs_loss
    Scheduler = paddle.optimizer.lr.MultiStepDecay(0.001, [adam_iter * 0.6, adam_iter * 0.8], gamma=0.1)
    Optimizer = paddle.optimizer.Adam(Scheduler)
    Optimizer.minimize(loss_batch)

    P_pre, U_pre, V_pre = model.out_transform(Val_inn)
    return [P_pre, U_pre, V_pre], [U_tra, V_tra], [data_loss, eqs_loss, loss_batch], Scheduler


def run_experiment(epoch_num, noise_type, noise, loss_type, weight, N=5000, _data=[], abnormal_size=0,
                   sieve_sigma=0, sieve_ratio=0):
    print(f'\n{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_{sieve_sigma}_{sieve_ratio}')
    try:
        import paddle.fluid as fluid
        place = fluid.CUDAPlace(0) if paddle.is_compiled_with_cuda() else fluid.CPUPlace()
    except:
        place = None

    paddle.enable_static()
    paddle.incubate.autograd.enable_prim()

    ## 数据生成
    # Domain bounds
    lb = np.array([1, -2, 0])
    ub = np.array([8, 2, 20])

    if len(_data) == 0:
        x_train, y_train, t_train, u_train, v_train, p_train = get_noise_data(N=N, noise_type=noise_type, sigma=noise,
                                                                              size=abnormal_size)
        _data.append((x_train, y_train, t_train, u_train, v_train))
    x_train, y_train, t_train, u_train, v_train = _data[0]
    Tra_DATA = np.concatenate(_data[0], axis=1).astype('float32')

    # Visualize the collocation points
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.scatter(_data[0][0].flatten(), _data[0][1].flatten(), marker='o', alpha=0.2, color='blue')
    plt.savefig('collocation.png')

    x, y, t, u, v, p = get_truth()
    Val_DATA = np.concatenate((x, y, t, u, v, p), axis=1).astype('float32')
    N_truth = Val_DATA.shape[0]

    ## 模型设置
    planes = [3] + [num_neurons] * num_layers + [2]
    # Model
    Net_model = PINN_NS_unsteady(planes=planes)

    ## 执行训练过程
    log_loss = []
    print_freq = args.print_freq
    save_freq = args.save_freq
    start_epoch = 0
    sta_time = time.time()

    if (sieve_sigma == 0) and (sieve_ratio == 0):
        if loss_type == 'l1':
            switch = 0
        else:
            switch = 1
        [P_pre, U_pre, V_pre], [U_tra, V_tra], Loss, Scheduler = build(Net_model, x_train, N_truth, weight, switch)

        exe = static.Executor(place)
        exe.run(static.default_startup_program())
        prog = static.default_main_program()

        for epoch in range(start_epoch, 1 + adam_iter):

            Scheduler.step()
            learning_rate = Scheduler.get_lr()
            train_items = exe.run(prog, feed={'Tra_inn': Tra_DATA[:, :3], 'Tra_out': Tra_DATA[:, 3:5],
                                              'Val_inn': Val_DATA[:, :3]},
                                  fetch_list=[[Net_model.get_lambda_1(), Net_model.get_lambda_2()] + Loss])
            lambda_1 = train_items[0][0]
            lambda_2 = train_items[1][0]
            loss = train_items[2:]
            if epoch > 0 and epoch % print_freq == 0:
                log_loss.append(np.array(loss).squeeze())
                print(
                    'epoch: {:6d}, lr: {:.3e}, data_loss: {:.3e}, pde_loss: {:.3e}, total_loss: {:.3e},  lambda_1: {:.3f}, lambda_2: {:.3f}, cost: {:.2f}'.
                    format(epoch, learning_rate, log_loss[-1][0], log_loss[-1][1], log_loss[-1][2], lambda_1, lambda_2,
                           time.time() - sta_time))
                plt.figure(100, figsize=(10, 6))
                plt.clf()
                plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'data_loss')
                plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'pde_loss', title='loss')
                plt.savefig(
                    path.joinpath(
                        f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_loss.png'))
            if epoch > 0 and epoch % save_freq == 0:
                all_items = exe.run(prog, feed={'Tra_inn': Tra_DATA[:, :3], 'Tra_out': Tra_DATA[:, 3:5],
                                                'Val_inn': Val_DATA[:, :3]},
                                    fetch_list=[[P_pre, U_pre, V_pre]])
                p_pre = all_items[0]
                u_pre = all_items[1]
                v_pre = all_items[2]
                paddle.save({'epoch': adam_iter, 'log_loss': log_loss, 'model': prog.state_dict()}, os.path.join(path,
                                                                                                                 f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_'
                                                                                                                 f'{sieve_sigma}_{sieve_ratio}.pdparams'))
                # paddle.save(prog, os.path.join(path, f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_'
                #                                      f'{weight}_{sieve_sigma}_{sieve_ratio}.pdmodel'))
                Net_model.plot_result(p_pre, u_pre, v_pre, path.joinpath(
                    f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_field.png'))

        data_items = exe.run(prog, feed={'Tra_inn': Tra_DATA[:, :3], 'Tra_out': Tra_DATA[:, 3:5],
                                         'Val_inn': Val_DATA[:, :3]},
                             fetch_list=[U_tra, V_tra])
        if loss_type == 'l1':
            log_ind = []
            for k in [2, 2.5, 3]:
                ind_rm = sieve_obs_sigma(Tra_DATA[:, 3:5], data_items[0], data_items[1], k=k)
                log_ind.append(ind_rm)
            for ratio in [0.1, 0.2, 0.3]:
                ind_rm = sieve_obs(Tra_DATA[:, 3:5], data_items[0], data_items[1], ratio=ratio)
                log_ind.append(ind_rm)
            np.save(f'data/{noise_type}.npy', log_ind)
    else:
        switch = 1
        [P_pre, U_pre, V_pre], [U_tra, V_tra], Loss, Scheduler = build(Net_model, x_train, N_truth, weight, switch)

        exe = static.Executor(place)
        exe.run(static.default_startup_program())
        prog = static.default_main_program()
        log_ind = np.load(f'data/{noise_type}.npy', allow_pickle=True)
        value = [2, 2.5, 3, 0.1, 0.2, 0.3]
        if sieve_sigma > 0 and sieve_ratio == 0:
            ind_rm = log_ind[value.index(sieve_sigma)]
            Tra_DATA[ind_rm, :] = 0
        elif sieve_sigma == 0 and sieve_ratio > 0:
            ind_rm = log_ind[value.index(sieve_ratio)]
            Tra_DATA[ind_rm, :] = 0
        else:
            print(sieve_sigma, sieve_ratio)
            raise Exception('not good')
        print("removed numbers:", len(Tra_DATA[ind_rm, :]))
        for epoch in range(start_epoch, 1 + adam_iter):

            Scheduler.step()
            learning_rate = Scheduler.get_lr()
            train_items = exe.run(prog, feed={'Tra_inn': Tra_DATA[:, :3], 'Tra_out': Tra_DATA[:, 3:5],
                                              'Val_inn': Val_DATA[:, :3]},
                                  fetch_list=[[Net_model.get_lambda_1(), Net_model.get_lambda_2()] + Loss])
            lambda_1 = train_items[0][0]
            lambda_2 = train_items[1][0]
            loss = train_items[2:]
            if epoch > 0 and epoch % print_freq == 0:
                log_loss.append(np.array(loss).squeeze())
                print(
                    'epoch: {:6d}, lr: {:.3e}, data_loss: {:.3e}, pde_loss: {:.3e}, total_loss: {:.3e},  lambda_1: {:.3f}, lambda_2: {:.3f}, cost: {:.2f}'.
                    format(epoch, learning_rate, log_loss[-1][0], log_loss[-1][1], log_loss[-1][2], lambda_1, lambda_2,
                           time.time() - sta_time))
                plt.figure(100, figsize=(10, 6))
                plt.clf()
                plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'data_loss')
                plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'pde_loss', title='loss')
                plt.savefig(
                    path.joinpath(
                        f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_{sieve_sigma}_{sieve_ratio}_loss.png'))
            if epoch > 0 and epoch % save_freq == 0:
                all_items = exe.run(prog, feed={'Tra_inn': Tra_DATA[:, :3], 'Tra_out': Tra_DATA[:, 3:5],
                                                'Val_inn': Val_DATA[:, :3]},
                                    fetch_list=[[P_pre, U_pre, V_pre]])
                p_pre = all_items[0]
                u_pre = all_items[1]
                v_pre = all_items[2]
                paddle.save({'epoch': adam_iter, 'log_loss': log_loss, 'model': prog.state_dict()}, os.path.join(path,
                                                                                                                 f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_'
                                                                                                                 f'{sieve_sigma}_{sieve_ratio}.pdparams'))
                # paddle.save(prog, os.path.join(path, f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_'
                # f'{weight}_{sieve_sigma}_{sieve_ratio}.pdmodel'))
                Net_model.plot_result(p_pre, u_pre, v_pre, path.joinpath(
                    f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_{sieve_sigma}_{sieve_ratio}_field.png'))

    paddle.save({'epoch': adam_iter, 'log_loss': log_loss,
                 'model': prog.state_dict()},
                os.path.join(path,
                             f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_{sieve_sigma}_{sieve_ratio}.pdparams'))
    # paddle.save(prog, os.path.join(path,
    # f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_{sieve_sigma}_{sieve_ratio}.pdmodel'))
    error_u, error_v, error_vel, error_max, error_p, error_p1, lambda_1_error, lambda_2_error = Net_model.predict_error(
        p_pre,
        u_pre,
        v_pre,
        lambda_1,
        lambda_2)
    Net_model.plot_result(p_pre, u_pre, v_pre, path.joinpath(
        f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_{sieve_sigma}_{sieve_ratio}_field.png'))
    with open(path.joinpath('result.csv'), 'a+') as f:
        f.write(
            f"{epoch_num},{loss_type},{N},{noise_type},{noise},{abnormal_size},{weight},{sieve_sigma},{sieve_ratio},{error_u},{error_v},{error_vel},{error_p},{error_p1},{lambda_1_error}, {lambda_2_error}\n")

    print("--- %s seconds ---" % (time.time() - sta_time))


def get_last_idx(filename):
    if not os.path.exists(filename):
        return -1
    with open(filename, "r") as f1:
        last_idx = int(f1.readlines()[-1].strip().split(',')[0])
        return last_idx


if __name__ == "__main__":
    idx = 0
    last_idx = get_last_idx(path.joinpath('result.csv'))
    executed_flag = False
    if executed_flag:
        sys.exit()
    for r in range(repeat):
        for noise_type in ['t1', 'none', 'normal', 'contamined']:
            for noise in [0.10]:
                for abnormal_ratio in [0]:
                    for N in [5000]:
                        # _data= []
                        for loss_type, sieve_sigma, sieve_ratio in zip(
                                ['l1', 'square', 'square', 'square', 'square', 'square', 'square', 'square', ],
                                [0, 0, 0, 0, 0, 2, 2.5, 3, ],
                                [0, 0, 0.1, 0.2, 0.3, 0, 0, 0, ]):
                            abnormal_size = int(N * abnormal_ratio)
                            if executed_flag:
                                sys.exit()
                            for weight in [1E-0]:
                                if idx > last_idx:
                                    run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=1,
                                                   loss_type=loss_type, abnormal_size=abnormal_size,
                                                   sieve_sigma=sieve_sigma, sieve_ratio=sieve_ratio)
                                    # executed_flag = True
                                idx += 1
    if executed_flag:
        sys.exit()

    for r in range(repeat):
        for noise_type in ['outlinear']:
            for noise, abnormal_ratio in zip([0.10], [0.10]):
                for N in [5000]:
                    for loss_type, sieve_sigma, sieve_ratio in zip(
                            ['l1', 'square', 'square', 'square', 'square', 'square', 'square', 'square'],
                            [0, 0, 0, 0, 0, 2, 2.5, 3],
                            [0, 0, 0.1, 0.2, 0.3, 0, 0, 0]):
                        abnormal_size = int(N * abnormal_ratio)
                        if executed_flag:
                            sys.exit()
                        for weight in [1E-0]:
                            if idx > last_idx:
                                run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=1,
                                               loss_type=loss_type, abnormal_size=abnormal_size,
                                               sieve_sigma=sieve_sigma, sieve_ratio=sieve_ratio)
                                # executed_flag = True
                            idx += 1
    if executed_flag:
        sys.exit()
    for r in range(repeat):
        for noise_type in ['outlinear']:
            for noise in [0.0]:
                for abnormal_ratio in [0.10]:
                    for N in [5000]:
                        for loss_type, sieve_sigma, sieve_ratio in zip(
                                ['l1', 'square', 'square', 'square', 'square', 'square', 'square', 'square'],
                                [0, 0, 0, 0, 0, 2, 2.5, 3],
                                [0, 0, 0.1, 0.2, 0.3, 0, 0, 0]):
                            abnormal_size = int(N * abnormal_ratio)
                            if executed_flag:
                                sys.exit()
                            for weight in [1E-0]:
                                if idx > last_idx:
                                    run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=1,
                                                   loss_type=loss_type, abnormal_size=abnormal_size,
                                                   sieve_sigma=sieve_sigma, sieve_ratio=sieve_ratio)
                                    # executed_flag = True
                                idx += 1


