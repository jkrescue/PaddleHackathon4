import matplotlib.pyplot as plt
import numpy as np
import paddle
from julia.api import Julia
from paddle.incubate.optimizer.functional import minimize_lbfgs
from paddle.io import DataLoader

jl = Julia(compiled_modules=False)
from dataloader import CustomDataset
from julia_coupling import JuliaRun
from model import AutoEncoder

test_data = CustomDataset(file_path="data/gaussian_train.npz", data_type="test")
test_loader = DataLoader(test_data, batch_size=128, shuffle=False, drop_last=True)

latent_dim_list = [100, 200, 400]
hidden_dim_list = latent_dim_list * 5

julia_module = JuliaRun()

indices = np.linspace(start=1, stop=10000, num=10000, dtype=int).reshape([100, 100])
obsindices = indices[range(16, 100, 17), :][:, range(16, 100, 17)].reshape([-1])
p_indices = np.linspace(start=1, stop=10000, num=10000, dtype=int).reshape([100, 100])
p_obsindices = indices[range(16, 100, 17), :][:, range(16, 100, 17)].reshape([-1])


def p2z(model, p):
    p_normalized = test_data.scaler.transform(p)
    mu, log_sigma = model.encoder(p_normalized)
    return mu


def z2p(model, z):
    p_normalized = model.decoder(z)
    return test_data.scaler.inverse_transform(p_normalized)


def objfunc(p, p_true):
    head = julia_module.func(p)
    head_true = julia_module.func(p_true)
    # return (1e2 * paddle.sum(paddle.pow(head[obsindices] - head_true[obsindices], 2))
    #         + 3e0 * paddle.sum(paddle.pow(p_true.flatten()[p_obsindices] - p.flatten()[p_obsindices], 2)))

    return (paddle.sum(paddle.pow(head - head_true, 2))
            + paddle.sum(paddle.pow(p_true - p, 2)))


fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 12))

for (i, sample) in enumerate(test_data.val_data):
    sample = paddle.to_tensor(sample).reshape([100, 100])
    axes[i, 0].imshow(sample.cpu().numpy(), cmap="jet", extent=[-50, 50, -50, 50], origin="lower",
                      interpolation="nearest")
    axes[i, 0].set_title("Reference Field")

    for nz_idx, (latent_dim, hidden_dim) in enumerate(zip(latent_dim_list, hidden_dim_list)):
        print("sample", i, "nz", latent_dim)

        p_model = AutoEncoder(input_dim=10000, latent_dim=latent_dim, hidden_dim=hidden_dim)
        model_save_path = "params_vae_nz{}/model.pdparams".format(latent_dim)
        p_model.set_state_dict(paddle.load(model_save_path))

        mu_stack = []
        with paddle.no_grad():
            for _, data_item in enumerate(test_loader()):
                mu, log_sigma = p_model.encoder(data_item)  # mu [B,latent_dim]
                mu_stack.append(mu)

        mean_latent = paddle.mean(paddle.concat(mu_stack, axis=0), axis=0, keepdim=True)
        cov_latent = paddle.linalg.cov(paddle.concat(mu_stack, axis=0), rowvar=False, ddof=False)


        def objfunc_z(z):
            p = z2p(p_model, z.reshape([1] + z.shape)).reshape(sample.shape)
            return ((paddle.mean((z - mean_latent) * (z - mean_latent)))
                    + objfunc(p, p_true=sample))


        z_init = mean_latent.flatten()
        # z_init = (paddle.randn([latent_dim]) - test_data.scaler.mean) / test_data.scaler.std
        z_minimizer = minimize_lbfgs(objective_func=objfunc_z, initial_position=z_init, initial_step_length=1e-2,
                                     max_iters=150)[2]
        p_pred = z2p(p_model, z_minimizer.reshape([1] + z_minimizer.shape)).reshape(sample.shape)

        relative_error = (paddle.sum(paddle.pow(p_pred - sample, 2)) /
                          paddle.sum(paddle.pow(paddle.mean(sample) - sample, 2)))
        print("relative_error:,", relative_error.item())

        im = axes[i, nz_idx + 1].imshow(p_pred.cpu().numpy(), cmap="jet", extent=[-50, 50, -50, 50], origin="lower",
                                        interpolation="nearest")
        axes[i, nz_idx + 1].set_title(f"nz=%d, relative_error:%f" % (latent_dim, relative_error.item()))

        np.savez("data-idx%d-nz%d.npz" % (i, nz_idx), sample=sample.cpu().numpy(), pred=p_pred.cpu().numpy())

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()
