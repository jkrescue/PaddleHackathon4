import paddle
import paddle.nn as nn


class AutoEncoder(paddle.nn.Layer):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(AutoEncoder, self).__init__()

        # encoder
        self._encoder_linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self._encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self._encoder_log_sigma = nn.Linear(hidden_dim, latent_dim)

        self._decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encoder(self, x):
        h = self._encoder_linear(x)
        mu = self._encoder_mu(h)
        log_sigma = self._encoder_log_sigma(h)
        return mu, log_sigma

    def decoder(self, x):
        return self._decoder(x)

    @staticmethod
    def kl_loss(mu, log_sigma):
        # 计算mu，log_sigma与 N(0,1)分布的差距
        base = paddle.exp(2. * log_sigma) + paddle.pow(mu, 2) - 1. - 2. * log_sigma
        loss = 0.5 * paddle.sum(base) / mu.shape[0]
        return loss

    def forward(self, x, noise):
        mu, log_sigma = self.encoder(x)
        z = mu + noise * paddle.exp(log_sigma)
        return mu, log_sigma, self.decoder(z)
