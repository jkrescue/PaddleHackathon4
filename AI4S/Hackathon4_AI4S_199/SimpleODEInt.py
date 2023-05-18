import paddle
import paddle.nn as nn
# import torch
# import torch.nn as nn
from enum import Enum

class Perturb(Enum):
    NONE = 0
    PREV = 1
    NEXT = 2


# Precompute divisions
_one_third = 1 / 3
_two_thirds = 2 / 3
_one_sixth = 1 / 6

def rk4_alt_step_func(func, t0, dt, t1, y0, f0=None, perturb=False):
    """Smaller error with slightly more compute."""
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0, perturb=Perturb.NEXT if perturb else Perturb.NONE)
    k2 = func(t0 + dt * _one_third, y0 + dt * k1 * _one_third)
    k3 = func(t0 + dt * _two_thirds, y0 + dt * (k2 - k1 * _one_third))
    k4 = func(t1, y0 + dt * (k1 - k2 + k3), perturb=Perturb.PREV if perturb else Perturb.NONE)
    return (k1 + 3 * (k2 + k3) + k4) * dt * 0.125


class SimpleFixedGridODESolver:
    # def __init__(self, func, y0, step_size=None, grid_constructor=None, interp="linear", perturb=False, **unused_kwargs):
    def __init__(self, func, y0, interp="linear", perturb=False):
        # self.atol = unused_kwargs.pop('atol')
        # unused_kwargs.pop('rtol', None)
        # unused_kwargs.pop('norm', None)
        # # _handle_unused_kwargs(self, unused_kwargs)
        # # del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        # self.device = y0.device
        # self.step_size = step_size
        self.interp = interp
        self.perturb = perturb

        # if step_size is None:
        #     if grid_constructor is None:
        #         self.grid_constructor = lambda f, y0, t: t
        #     else:
        #         self.grid_constructor = grid_constructor
        # else:
        #     if grid_constructor is None:
        #         self.grid_constructor = self._grid_constructor_from_step_size(step_size)
        #     else:
        #         raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    def _step_func(self, func, t0, dt, t1, y0):
        pass
    
    def integrate(self, t):
        # time_grid = self.grid_constructor(self.func, self.y0, t)
        # assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = paddle.empty([len(t), *self.y0.shape], dtype=self.y0.dtype)
        # solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device) #torch
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        # for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
        for t0, t1 in zip(t[:-1], t[1:]):
            dt = t1 - t0
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy
            
            solution[j] = y1

            while j < len(t) and t1 >= t[j]:
                if self.interp == "linear":
                    solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    solution[j] = self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t[j])
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                j += 1
            y0 = y1

        return solution
    
    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)


class RK4(SimpleFixedGridODESolver):
    order = 4

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return rk4_alt_step_func(func, t0, dt, t1, y0, f0=f0, perturb=self.perturb), f0

SOLVERS = {
    # 'dopri8': Dopri8Solver,
    # 'dopri5': Dopri5Solver,
    # 'bosh3': Bosh3Solver,
    # 'fehlberg2': Fehlberg2,
    # 'adaptive_heun': AdaptiveHeunSolver,
    # 'euler': Euler,
    # 'midpoint': Midpoint,
    'rk4': RK4,
    # 'explicit_adams': AdamsBashforth,
    # 'implicit_adams': AdamsBashforthMoulton,
    # # Backward compatibility: use the same name as before
    # 'fixed_adams': AdamsBashforthMoulton,
    # # ~Backwards compatibility
    # 'scipy_solver': ScipyWrapperODESolver,
}


# class _StitchGradient(paddle.autograd.Function):
#     @staticmethod
#     def forward(ctx, x1, out):
#         return out

#     @staticmethod
#     def backward(ctx, grad_out):
#         return grad_out, None


# def _nextafter(x1, x2):
#     with paddle.no_grad():
#         if hasattr(torch, "nextafter"):
#             out = paddle.nextafter(x1, x2)
#         else:
#             out = np_nextafter(x1, x2)
#     return _StitchGradient.apply(x1, out)


class _PerturbFunc(nn.Layer):
# class _PerturbFunc(nn.Module):
    
    def __init__(self, base_func):
        super(_PerturbFunc, self).__init__()
        self.base_func = base_func

    def forward(self, t, y, *, perturb=Perturb.NONE):
        assert isinstance(perturb, Perturb), "perturb argument must be of type Perturb enum"
        # This dtype change here might be buggy.
        # The exact time value should be determined inside the solver,
        # but this can slightly change it due to numerical differences during casting.
        t = paddle.cast(t, y.dtype)
        # t = t.to(y.dtype) # torch
        # if perturb is Perturb.NEXT:
        #     # Replace with next smallest representable value.
        #     t = _nextafter(t, t + 1)
        # elif perturb is Perturb.PREV:
        #     # Replace with prev largest representable value.
        #     t = _nextafter(t, t - 1)
        # else:
        #     # Do nothing.
        #     pass
        return self.base_func(t, y)

def simple_odeint(func, y0, t, method=None):
    func = _PerturbFunc(func)
    
    solver = SOLVERS[method](func=func, y0=y0)
    solution = solver.integrate(t)
    
    return solution
    
    

