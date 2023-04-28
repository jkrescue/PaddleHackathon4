import paddle


def flatten(x):
    return x.flatten()


def unflatten(x):
    return x.reshape((2,2))


# 假设网络参数超过一维
def net(x):
    assert len(x.shape) > 1
    return x.square().mean()


# 待优化函数
def bfgs_f(flatten_x):
    return net(unflatten(flatten_x))


x = paddle.rand([2,2])
for i in range(0, 10):
    # 使用 minimize_lbfgs 前，先将 x 展平
    x_update = paddle.incubate.optimizer.functional.minimize_lbfgs(bfgs_f, flatten(x))[2]
    # 将 x_update unflatten，然后更新参数
    paddle.assign(unflatten(x_update), x)