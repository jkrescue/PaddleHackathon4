import paddle
import paddle.nn as nn
import os
import torch.nn.functional as F

class DeepModelMulti(nn.Layer):
    """
    PINNS for mulit-networks
    """

    # =============================================================================
    #     Inspired by Haghighat Ehsan, et all.
    #     "A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics"
    #     Computer Methods in Applied Mechanics and Engineering.
    # =============================================================================
    def __init__(self, planes, active=nn.GELU()):
        """
        :param planes: list，[M,...,N],全连接神经网络的输入维度M，每个隐含层维度，输出维度N
        :param active: 激活函数
               与single区别，multi采用N个全连接层,每个全连接层输出维度为1
        """

        super(DeepModelMulti, self).__init__()
        self.planes = planes
        self.active = active

        self.layers = nn.LayerList()
        for j in range(self.planes[-1]):
            layer = []
            for i in range(len(self.planes) - 2):
                layer.append(nn.Linear(self.planes[i], self.planes[i + 1], weight_attr=nn.initializer.XavierNormal()))
                layer.append(self.active)
            layer.append(nn.Linear(self.planes[-2], 1, weight_attr=nn.initializer.XavierNormal()))
            self.layers.append(nn.Sequential(*layer))

    def forward(self, in_var):
        """
        神经网络前向计算
        """
        y = []
        for i in range(self.planes[-1]):
            y.append(self.layers[i](in_var))
        return paddle.concat(y, axis=-1)

    def loadmodel(self, File):
        """
        神经网络模型权重读入
        """
        try:
            checkpoint = paddle.load(File)
            self.set_state_dict(checkpoint['model'])  # 从字典中依次读取
            start_epoch = checkpoint['epoch']
            print("load start epoch at epoch " + str(start_epoch))
        except:
            print("load model failed！ start a new model.")

        return

    def equation(self, **kwargs):
        """
        约束方程损失
        """
        return 0

class DeepModelSingle(nn.Layer):
    """
    PINNS for mulit-networks
    """

    # =============================================================================
    #     Inspired by M. Raissi a, P. Perdikaris b,∗, G.E. Karniadakis.
    #     "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems
    #     involving nonlinear partial differential equations".
    #     Journal of Computational Physics.
    # =============================================================================
    def __init__(self, planes, active=nn.GELU()):
        """
        :param planes: list，[M,...,N],全连接神经网络的输入维度，每个隐含层维度，输出维度
        :param active: 激活函数
                       与multi相比，single采用1个全连接层,该全连接层输出维度为N
        """
        super(DeepModelSingle, self).__init__()
        self.planes = planes
        self.active = active

        self.layers = nn.LayerList()
        for i in range(len(self.planes) - 2):
            self.layers.append(nn.Linear(self.planes[i], self.planes[i + 1], weight_attr=nn.initializer.XavierNormal()))
            self.layers.append(self.active)
        self.layers.append(nn.Linear(self.planes[-2], self.planes[-1], weight_attr=nn.initializer.XavierNormal()))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, in_var):
        """
        神经网络模型前向计算
        """
        out_var = self.layers(in_var)
        return out_var

    def loadmodel(self, File):
        """
        神经网络模型权重读入
        """
        try:
            checkpoint = paddle.load(File)
            self.set_state_dict(checkpoint['model'])  # 从字典中依次读取
            start_epoch = checkpoint['epoch']
            print("load start epoch at epoch " + str(start_epoch))
        except:
            print("load model failed！ start a new model.")

    def equation(self, **kwargs):
        """
        约束方程损失
        """
        return 0

class DeepONetSingle(nn.Layer):
    """
    deeponet for multi-input
    """

    # =============================================================================
    #     Inspired by L. Lu, J. Pengzhan, G.E. Karniadakis.
    #     "DeepONet: Learning nonlinear operators for identifying differential equations based on
    #     the universal approximation theorem of operators".
    #     arXiv:1910.03193v3 [cs.LG] 15 Apr 2020.
    # =============================================================================
    def __init__(self, planes_branch, planes_trunk, nums_branch, active=nn.GELU()):
        """
        :param planes_branch: list，[M,...,N],branch_net全连接神经网络的输入维度，每个隐含层维度，输出维度
        :param planes_trunk: list，[M,...,N],trunk_net全连接神经网络的输入维度，每个隐含层维度，输出维度
        :param active: 激活函数
                       与multi相比，single采用1个全连接层,该全连接层输出维度为N
        """
        super(DeepONetSingle, self).__init__()
        self.branch = [DeepModelSingle(planes_branch, active=active)] * nums_branch
        self.trunk = DeepModelSingle(planes_trunk, active=active)

    def forward(self, u_var, y_var):
        """
        神经网络模型前向计算
        """
        Bs = [branch(u) for branch, u in zip(self.branch, u_var)]
        T = self.trunk(u_var, y_var)
        Rs = paddle.stack([B * T for B in Bs], axis=-1).sum(dim=-1)
        out_var = paddle.sum(Rs, axis=-1)
        return out_var

    def loadmodel(self, File):
        """
        神经网络模型权重读入
        """
        try:
            checkpoint = paddle.load(File)
            self.set_state_dict(checkpoint['model'])  # 从字典中依次读取
            start_epoch = checkpoint['epoch']
            print("load start epoch at epoch " + str(start_epoch))
        except:
            print("load model failed！ start a new model.")

    def equation(self, **kwargs):
        """
        约束方程损失
        """
        return 0

class DeepONetMulti(nn.Layer):
    def __init__(self, input_dim: int, operator_dims: list, output_dim: int,
                 planes_branch: list, planes_trunk: list, active=nn.GELU()):
        """
        :param input_dim: int, the coordinates dim for trunk net
        :param operator_dims: list，the operate dims list for each branch net
        :param output_dim: int, the predicted variable dims
        :param planes_branch: list, the hidden layers dims for branch net
        :param planes_trunk: list, the hidden layers dims for trunk net
        :param activation: activation function
        """
        super(DeepONetMulti, self).__init__()
        self.branches = nn.LayerList()
        self.trunks = nn.LayerList()
        for dim in operator_dims:
            self.branches.append(DeepModelSingle([dim] + planes_branch, active=active))
        for _ in range(output_dim):
            self.trunks.append(DeepModelSingle([input_dim] + planes_trunk, active=active))

        self.reset_parameters()

    def reset_parameters(self):
        """
        weight initialize
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight, gain=1)
                nn.initializer.XavierNormal(m.weight)
                m.bias.data.zero_()

    def forward(self, u_vars, y_var):
        """
        forward compute
        :param in_var: (batch_size, ..., input_dim)
        """
        B = 1.
        for u_var, branch in zip(u_vars, self.branches):
            B *= branch(u_var)
        out_var = []
        for trunk in self.trunks:
            T = trunk(y_var)
            out_var.append(paddle.sum(B * T, axis=-1))
        out_var = paddle.stack(out_var, axis=-1)
        return out_var

    def loadmodel(self, File):
        """
        神经网络模型权重读入
        """
        try:
            checkpoint = paddle.load(File)
            self.set_state_dict(checkpoint['model'])  # 从字典中依次读取
            start_epoch = checkpoint['epoch']
            print("load start epoch at epoch " + str(start_epoch))
        except:
            print("load model failed！ start a new model.")

    def equation(self, **kwargs):
        """
        约束方程损失
        """
        return 0

