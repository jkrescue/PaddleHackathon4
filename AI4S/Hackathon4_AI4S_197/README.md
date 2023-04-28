# Paddle Hackathon 第4期 科学计算—— 科学计算方向 197
 
## 1.背景
原文：[Robust Regression with Highly Corrupted Data via PhysicsInformed Neural Networks](https://arxiv.org/abs/2210.10646)

参考：[robust_pinn](https://github.com/weipengOO98/robust_pinn)

- 近年来，大量文献研究表明PINN可以基于少量数据和PDE方程解决各类正向和反向问题。但是，当测量数据因仪器等原因误差较大时，求解精度无法保证。因为最小二乘法在数据偏差较大时会放大偏差，该文章提出用LAD-PINN(数据损失函数基于L1范数的PINN)来求解数据异常下的PDE和未知参数，并提出了基于中值绝对偏差的两阶段PINN方法(MAD-PINN)，以提高求解结果的准确性。
- 在MAD-PINN中，LAD-PINN作为异常数据检测器用于一次求解，然后通过中指绝对偏差筛选出偏差较大的数据，基于剩余正常数据，以L2范数为损失函数的PINN（OLS-PINN）对PDE进行重新求解。
- 在数据异常检测中，作者提供了两种数据清理准则：
    - 以最大误差为基准固定百分比，设置数据清理偏差范围
    - 假设数据误差服从正态分布，利用MAD估计标准差，并用其构造一个阈值来排除异常值：
    $$\hat{\partial}_c=\frac{1}{1.6777}+\operatorname{median}\left(\left|u_i-\hat{a}^{+}\left(t_i, x_i\right)\right|\right)$$
- 文章利用四个问题：泊松方程、波动方程、稳态和非稳态的NS方程进行方法验证。

## 2.代码说明
### 额外库需求
- paddle
- pyDOE

###  相关运行结果[Robust_PINNs4paddle AI studio](https://aistudio.baidu.com/aistudio/projectdetail/5864276)
  - 代码文件

    - poisson/* 论文Section 4.1. 泊松方程
    - piv/* 论文Section 4.2. 稳态二维圆柱绕流（层流）
    - wave/* 论文Section 4.3. 一维波动方程
    - ns/* 论文Section 4.4. 非稳态二维圆柱绕流
      - unsteady_NS.py为一阶段运行脚本
      - unsteady_NS_std.py中训练PINN两次，数据损失函数分别为L1和L2
      - unsteady_NS_two_stage.py为二阶段运行脚本
      - run_size.sh和unsteady_NS_noise.py分别执行两类对比循环
    - basic_model.py 中为实现的全连接神经网络结构
    - parser_pinn.py 中包含了配置参数
    - gen_*_data.py中包含生成真实场及异常数据的函数
  - fig文件夹包含与论文对比图片，序号与文献一一对应

### 数据库
- [steady NS](https://github.com/dsqzhou/paddle-pinn/blob/main/piv/FluentSol.mat)
- [unsteady NS](https://github.com/dsqzhou/paddle-pinn/blob/main/ns/cylinder_nektar_wake.mat)

### 快速开始
  * poisson文件夹中，poisson_eq.py包含pinn的搭建和训练，及得到相关变量损失和曲线图，存在demo可运行；分别运行*_noise.py/
*_size.py/*_weight.py可获得不同噪声程度、不同尺寸、不同损失权重下对应的不同噪声数据的模拟结果；运行*_one-outlier.py获得一个异常数据下
的预测结果
  * piv文件夹中，steady_NS.py包含pinn的搭建和训练，及得到相关变量损失和曲线图，存在demo可运行；分别运行*_noise.py/
*_size.py可获得不同噪声程度、不同尺寸下对应的不同噪声数据的模拟结果；运行*_two_stage.py获得两阶段PINN方法的预测结果。
的预测结果
  * wave文件夹中，wave_eq.py包含pinn的搭建和训练，及得到相关变量损失和曲线图，存在demo可运行；分别运行*_noise.py/
*_size.py/*_weight.py可获得不同噪声程度、不同尺寸、不同损失权重下对应的不同噪声数据的模拟结果
  * ns文件夹中，unsteady_NS.py包含pinn的搭建和训练，及得到相关变量损失和曲线图，存在demo可运行；分别运行unsteady_NS_noise.py/
*run_size.py可获得不同噪声程度、不同尺寸下对应的不同噪声数据的模拟结果；运行unsteady_NS_two_stage.py获得两阶段PINN方法的预测结果。
的预测结果

## 3.环境依赖
### 特别提示，Section4.4中用到了三阶微分，高阶自动微分需要在develop版本下静态图模式运行，安装以下版本
```
python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
export LD_LIBRARY_PATH=/opt/conda/envs/python35-paddle120-env/lib:${LD_LIBRARY_PATH}
```
- 同时，LBFGS优化器也需要在该版本下运行

## 计算结果
### 4.1 泊松方程
* 原始方程为1-D Poisson方程：

  $$u_{xx}=-16sin(4x) \in[-\pi, \pi]$$

* 方程解析解为：

  $$u(x)=sin(4x)+1$$

观测数据仅分布在 $[-\pi, 0]$ ，需确定 $[-\pi, \pi]$ 内的解
* Figure3-4证明了在无物理约束下，神经网络没有物理定律的泛化能力，无法依赖 $[-\pi, 0]$ 的数据学习到 $[0, \pi]$ 内的正确解。并且，针对噪声数据，神经网络在充足训练下会过度拟合数据。而加入PDE损失后，两种PINN求解结果都与实际基本吻合。
* Figure5显示了LAD与OLS再应对异常高值时的预测能力，在仅包含一个异常高值（x=0,y=10）的情况，OLS-PINN在 $x \in[0, \pi]$ 的预测能力相比较差，曲线略微倾斜于异常点，说明L2范数会放大个别大误差的影响，导致整体预测的偏颇。与论文相比，复现结果中OLS的预测效果明显更好。

|      |  复现  | 论文 |
|:--------------:| :------------: | :------: |
|Figure3 | ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/Figure3.png)|![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/fig3.png) |
|Figure4|![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/Figure4.png)|![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/fig4.png)|
|Figure5|![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/Figure5.png)|![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/fig5.png)|

* 下表详细展示了采用LAD-PINN和OLS-PINN对1D Possion问题的预测效果，以及不同数据类型、不同损失程度、不同测点数量和不同PDE损失权重对于两种PINN的影响。其中，左侧为本次复现结果，而右侧为论文结果。

|      |                                     复现                                     | 论文 |
|:------------:|:--------------------------------------------------------------------------:| :------: |
|Table2 |   ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/tab2.png)   |![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/tab2.png) |
|Table3| ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/tab3.png) |![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/tab3.png)|
|Figure6| ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/fig6.png) |![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/fig6.png)|
### 4.2 二维稳态NS方程
#### 问题描述
经典二维稳态圆柱绕流问题
* NS方程（层流）：

$$
\begin{aligned}
& u u_x+v u_y=-p_x+u_{x x}+u_{y y} \\
& u v_x+v v_y=-p_y+v_{x x}+v_{y y}
\end{aligned}
$$
   
* 连续性方程：

  $$u_x+ v_y=0$$
  
[](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/fig7.png)
几何结构如上图所示，红线代表壁面。密度 $\rho=1 \mathrm{~kg} / \mathrm{m}^3$ ，动力粘度 $\mu=2 \cdot 10^{-2} \mathrm{~kg} / \mathrm{m}^3$ 
* 边界条件：壁面无滑移条件，速度为0；出口压力为0；入口速度设置为： $u(0, y)=4 \frac{U_M}{H^2}(H-y) y$ ，其中 $U_M=1 \mathrm{m} / \mathrm{s}$ ， $H=0.41 \mathrm{m}$

* 所给数据集仅包含（x,y,u,v），压力未知，入口速度边界未知。
* 论文假设存在流函数 $\psi(x, y)$ ，使得 $u=\psi_y, \quad v=-\psi_x$ ，从而自动满足连续性方程。并且引入柯西应力张量 $\sigma$ 来降低方程中导数阶数：

$$
\begin{aligned}
\sigma^{11} & =-p+2 \mu u_x \\
\sigma^{22} & =-p+2 \mu v_y \\
\sigma^{12} & =\mu\left(u_y+v_x\right) \\
p & =-\frac{1}{2}\left(\sigma^{11}+\sigma^{22}\right) \\
\left(u u_x+v u_y\right) & =\mu\left(\sigma_x^{11}+\sigma_y^{12}\right) \\
\left(u v_x+v v_y\right) & =\mu\left(\sigma_x^{12}+\sigma_y^{22}\right)
\end{aligned}
$$

* 建立神经网络： $\psi, p, \sigma^{11}, \sigma^{12}, \sigma^{22}=\operatorname{net}(x, y)$ 两输入、五输出
paddle代码实现如下：
```
class PINN_laminar_flow(DeepModelSingle):
    def __init__(self, planes):
        super(PINN_laminar_flow, self).__init__(planes, active=nn.Tanh())
        self.rho = 1.0
        self.mu = 0.02

    def gradients(self, y, x):
        return paddle.grad(y, x, grad_outputs=paddle.ones_like(y),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    def equation(self, inn_var, out_var):
        p = out_var[..., 0:1]
        psi = out_var[..., 1:2]
        s11 = out_var[..., 2:3]
        s22 = out_var[..., 3:4]
        s12 = out_var[..., 4:5]
        u = self.gradients(psi, inn_var)[..., 1:2]
        v = -self.gradients(psi, inn_var)[..., 0:1]

        dpda = self.gradients(p, inn_var)
        duda = self.gradients(u, inn_var)
        dvda = self.gradients(v, inn_var)
        ds11da = self.gradients(s11, inn_var)
        ds22da = self.gradients(s22, inn_var)
        ds12da = self.gradients(s12, inn_var)

        dpdx, dpdy = dpda[..., 0:1], dpda[..., 1:2]
        dudx, dudy = duda[..., 0:1], duda[..., 1:2]
        dvdx, dvdy = dvda[..., 0:1], dvda[..., 1:2]
        ds11dx, ds11dy = ds11da[..., 0:1], ds11da[..., 1:2]
        ds22dx, ds22dy = ds22da[..., 0:1], ds22da[..., 1:2]
        ds12dx, ds12dy = ds12da[..., 0:1], ds12da[..., 1:2]

        res_u = self.rho * (u * dudx + v * dudy) - ds11dx - ds12dy
        res_v = self.rho * (u * dvdx + v * dvdy) - ds12dx - ds22dy
        # res_c = dudx + dvdy
        res_s11 = -p + 2 * self.mu * dudx - s11
        res_s22 = -p + 2 * self.mu * dvdy - s22
        res_s12 = self.mu * (dudy + dvdx) - s12
        res_p = p + (s11 + s22) / 2

        return paddle.concat((res_u, res_v, res_s11, res_s22, res_s12, res_p), axis=-1), \
               paddle.concat((p, u, v), axis=-1)
```
- 损失函数包含三部分：速度数据对比损失，边界损失和方程残差损失
#### 计算结果

|      |                                     复现                                     | 论文 |
|:------------:|:--------------------------------------------------------------------------:| :------: |
|Figure8 | ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/fig8.png) |![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/fig8.png) |
|Figure9| ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/fig9.png) |![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/fig9.png)|
### 4.3 波动方程
#### 问题描述
一维波动方程（逆向求解参数c）

$$u_{t t}=c \cdot u_{x x}, \quad(t, x) \in \Omega=[0,2 \pi] \times[0, \pi]$$

[](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/wave_u-xt.png)
[](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/wave_u-t.png)
解析解：

$$u=\sin x \cdot(\sin \sqrt{c} \cdot t+\cos \sqrt{c} \cdot t)$$

原文描述：c=1，实际 $\sqrt{c}=1.54$ 

|  c=1  | $\sqrt{c}=1.54$ |
| :------------: | :------: |
|![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/wave_c%3D1.png)|![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/wave_c%3D1.54**2.png)|

建立神经网络： $u=\operatorname{net}(x, t)$ 两输入、一输出
paddle代码实现如下：
```
class PINN_wave(DeepModelSingle):
    def __init__(self, planes):
        super(PINN_wave, self).__init__(planes, active=nn.Tanh())
        self.c = paddle.fluid.layers.create_parameter(shape=[1], dtype='float32')

    def gradients(self, y, x):
        return paddle.grad(y, x, grad_outputs=paddle.ones_like(y),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    def equation(self, inn_var):
        model = self
        u = model(inn_var)
        duda = self.gradients(u, inn_var)

        dudx, dudt = duda[..., 0:1], duda[..., 1:2]
        d2udx2 = self.gradients(dudx, inn_var)[..., 0:1]
        d2udt2 = self.gradients(dudt, inn_var)[..., 1:2]

        res_u = d2udt2 - self.c * d2udx2

        return res_u
```
#### 计算结果

|          |                                       复现                                       |                                       论文                                        |
|:--------:|:------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------:|
|  Table7  |   ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/tab7.png)   | ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/tab7.png)  |
|  Table8  |   ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/tab8.png)   | ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/tab8.png)  |
|  Table9  |   ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/tab9.png)   | ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/tab9.png)  |
| Table10  |  ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/tab10.png)   | ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/tab10.png) |
| Figure15 | ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/Figure15.png) | ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/fig15.png) |

### 4.4 含有未知参数的二维非稳态NS方程
#### 问题描述
不可压缩流体流动问题
* NS方程：

$$
\begin{aligned}
u_t+\lambda_1\left(u u_x+v u_y\right) & =-p_x+\lambda_2\left(u_{x x}+u_{y y}\right) \\
v_t+\lambda_1\left(u v_x+v v_y\right) & =-p_y+\lambda_2\left(v_{x x}+v_{y y}\right)
\end{aligned}
$$

* 连续性方程：

$$
u_x+ v_y=0
$$

* 论文假设存在流函数$\psi(x, y)$，使得 $u=\psi_y, \quad v=-\psi_x$ ，从而自动满足连续性方程。
* 建立神经网络： $\psi, p=\operatorname{net}(t, x, y, \lambda_1,  \lambda_2)$ ，与问题4.2相比，该神经网络只有两个方程约束，但同时要求三阶导数。
方程paddle代码实现如下：
```
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
```
研究问题：圆柱绕流，假设无量纲自由流速度 $u_{\infty}=1$ ，圆柱体直径D =1，运动粘度ν=0.01，雷诺数Re=100，即： $\lambda_1 = 1，\lambda_2 = 0.01$ 。系统表现出周期性稳态行为，其特征为圆柱体尾迹中不对称的涡脱模式，称为卡门涡街。

![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/NS_field.png)
![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/Figure16.png)

几何结构如上图所示，时空训练数据采样于圆柱体后面的矩形区。蓝色点表示速度分量u(t, x, y)、v(t, x, y)的训练数据点位置。
* 所给数据集包含20s内该矩形域下的速度场分布（ $x\in [1,8], y\in [-2,2], t\in [0,20]$ ），压力未知，无边界条件。
- 损失函数包括两部分：方程损失+速度数据损失
#### 计算结果
LAD-PINN与OLS-PINN对比

|          |                                       复现                                       |                                       论文                                        |
|:--------:|:------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------:|
| Table11  |  ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/tab11.png)   | ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/tab11.png) |
| Table12  |  ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/tab12.png)   | ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/tab12.png) |
| Table13  |  ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/tab13.png)   | ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/tab13.png) |
| Table14  |  ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/tab14.png)   | ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/tab14.png) |
二阶段方法压力重构
* 在未给参考压力下，所求压力值都是相对值，因此重构压力场会与参考数据相差一个常数值。这个常数可利用最小化平方范数表示：

$$
\hat{c}=\arg \min _c\left\|p_{r e f}-\hat{p}+c\right\|_2^2
$$

* 利用优化问题的一阶条件，可得

$$
\hat{c}:=\frac{\int_{\Omega}\left(\hat{p}-p_{r e f}\right) d x d y}{\int_{\Omega} d x d y}
$$

* 随后，本文采用相对平方误差来判断重构压力场的预测精度:

$$
\operatorname{rel}_p:=\frac{\left\|p_{\text {ref }}-\hat{p}+\hat{c}\right\|_2}{\left\|p_{r e f}\right\|_2}
$$

* 代码表现为：
```
t_p = 100
p_pred_slice = p_pred[t_p::200, 0]
p_slice = p[t_p::200, 0]
p_mean_slice = np.mean(p_pred_slice - p_slice)
error_p = np.linalg.norm(p_pred_slice - p_slice - p_mean_slice, 2) / np.linalg.norm(p_slice, 2)
```
- 下图对比了t=5/10/15s时不同PINN方法的压力误差情况，由于文献中未告知图片采用哪种异常数据计算的，在此放了clean和mixed-outlier两种情况。在无噪声下，OLS表现比LAD好，MAD表现最好，但与OLS的结果相差不大。在mixed-outlier中，OLS无法获得较准确的压力场，而LAD/MAD的误差都比较小，MAD的结果相对最好，与原文结果基本一致。

|      |                                     复现                                     | 论文 |
|:------------:|:--------------------------------------------------------------------------:| :------: |
|Figure17 none| ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/fig17_none.png) |![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/fig17.png)|
|Figure17 mixed| ![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/paddle/fig17_none.png) |![](https://github.com/dsqzhou/rPINN_paddle/blob/main/fig/literature/fig17.png)|
