import numpy as np


def get_truth():
    x = np.linspace(0, np.pi, 201)
    t = np.linspace(0, 2 * np.pi, 401)
    c = 1.54

    X, T = np.meshgrid(x, t)
    Exact = np.sin(X) * (np.sin(c * T) + np.cos(c * T))

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]
    return X_star, u_star


def get_noise_data(N, noise_type="none", sigma=0.5, size=-1, seed=0):
    np.random.seed(seed=seed)
    xx, yy = get_truth()
    idx = np.random.choice(xx.shape[0], N, replace=False)
    xx = xx[idx, :]
    yy = yy[idx, :]
    yy = yy.ravel()
    sigma = sigma * np.std(yy)
    if noise_type == "none":  # Clean
        return xx, yy[:, None]
    elif noise_type == "normal":  # Gaussian
        eps = np.random.randn(N) * sigma
        return xx, (yy + eps)[:, None]
    elif noise_type == "contamined":  # Contaminated, variance reduced GMM sampling
        eps = np.random.randn(N) * sigma
        idx = np.random.choice(np.arange(0, N), size=N // 5, replace=False)
        eps[idx] = np.random.randn(len(eps[idx])) * sigma * 10
        return xx, (yy + eps)[:, None]
    elif noise_type == "t1":  # Cauchy
        eps = np.random.standard_cauchy(N) * sigma
        return xx, (yy + eps)[:, None]
    elif noise_type == 'outlinear':  # Outlier or Mixed
        eps = np.random.randn(N) * sigma
        if size < 0:
            idx = np.random.choice(np.arange(0, N), size=N // 10, replace=False)
        else:
            idx = np.random.choice(np.arange(0, N), size=size, replace=False)
        yy += eps
        yy[idx] = 10
        return xx, yy[:, None]
    else:
        raise NotImplementedError(f'noise type {noise_type} not implemented.')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X_star_tru, u_star_tru = get_truth()
    shape = (401, 201)
    x_tru = X_star_tru[:, 0].reshape(*shape)
    t_tru = X_star_tru[:, 1].reshape(*shape)
    print(X_star_tru.shape)

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), dpi=200)
    fig.set_tight_layout(True)
    np.random.seed(0)
    X_star, u_star = get_noise_data(250, noise_type='normal')
    x, t = X_star[:, 0], X_star[:, 1]
    ax[0].scatter(t, u_star, label='Gaussian', marker='o')

    X_star, u_star = get_noise_data(250, noise_type='outlinear', sigma=0.1, size=1)
    x, t = X_star[:, 0], X_star[:, 1]
    ax[2].scatter(t, u_star, label='mixed')

    ax[1].pcolormesh(t_tru.ravel().reshape(*shape), x_tru.ravel().reshape(*shape), u_star_tru.ravel().reshape(*shape),
                     vmin=-1,
                     vmax=1.)
    arr = np.random.randint(0, 80601, size=250)
    ax[1].scatter(t_tru.ravel()[arr], x_tru.ravel()[arr], s=10, marker='*', c='black')
    ax[0].scatter(t_tru.ravel()[arr], u_star_tru.ravel()[arr], label='exact', marker='*')
    plt.legend()
    plt.savefig('noise_data_c=1.54.png')
    plt.show()
