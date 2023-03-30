import argparse
import os

import numpy as np
import paddle
import paddle.nn as nn
from paddle import optimizer
from paddle.io import DataLoader

from dataloader import CustomDataset
from model import AutoEncoder

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type=str, default='0')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=int, default=1e-4)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

print("load data start...")
train_data = CustomDataset(file_path="data/gaussian_train.npz", data_type="train")
test_data = CustomDataset(file_path="data/gaussian_train.npz", data_type="test")
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=True)
print("load data finished")


def train(latent_dim, hidden_dim):
    os.makedirs("params_vae_nz{}".format(latent_dim), exist_ok=True)

    criterion = nn.MSELoss()
    p_model = AutoEncoder(input_dim=10000, latent_dim=latent_dim, hidden_dim=hidden_dim)
    optim = optimizer.Adam(parameters=p_model.parameters(), learning_rate=args.lr)
    f = open("params_vae_nz{}/log.txt".format(latent_dim), "w+")
    min_test_losses = 1000000
    for epoch in range(args.epoch):
        train_losses = []
        for i, data_item in enumerate(train_loader()):
            '''data_item [B,F]'''
            noise = paddle.randn(shape=[args.batch_size, latent_dim])
            mu, log_sigma, decoder_z = p_model(data_item, noise)
            loss = p_model.kl_loss(mu, log_sigma) + criterion(decoder_z, data_item)
            train_losses.append(loss.item())

            loss.backward()
            optim.step()
            optim.clear_gradients()
        # print("epoch:", epoch, "train loss:", np.mean(train_losses), flush=True)
        f.write("epoch: " + str(epoch) + " train loss: " + str(np.mean(train_losses)) + "\n")

        with paddle.no_grad():
            test_losses = []
            for i, data_item in enumerate(test_loader()):
                noise = paddle.randn(shape=[args.batch_size, latent_dim])
                mu, log_sigma, decoder_z = p_model(data_item, noise)

                loss = p_model.kl_loss(mu, log_sigma) + criterion(decoder_z, data_item)
                test_losses.append(loss.item())
            # print("epoch:", epoch, "test loss:", np.mean(test_losses), flush=True)
            f.write("epoch: " + str(epoch) + " test loss: " + str(np.mean(test_losses)) + "\n")

            if np.mean(test_losses) < min_test_losses:
                model_save_path = "params_vae_nz{}/model.pdparams".format(latent_dim)
                optim_save_path = "params_vae_nz{}/adam.pdopt".format(latent_dim)
                os.makedirs("params_vae_nz{}".format(latent_dim), exist_ok=True)
                paddle.save(p_model.state_dict(), model_save_path)
                paddle.save(optim.state_dict(), optim_save_path)
                min_test_losses = np.mean(test_losses)

        f.flush()
    f.close()


if __name__ == '__main__':
    paddle.seed(1)

    latent_dim_list = [100, 200, 400]
    hidden_dim_list = latent_dim_list * 5
    for (latent_dim, hidden_dim) in zip(latent_dim_list, hidden_dim_list):
        train(latent_dim, hidden_dim)
