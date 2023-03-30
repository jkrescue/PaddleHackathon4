import paddle
# julieries
from julia.api import Julia

jl = Julia(compiled_modules=False)
# import julia
from julia import Main


def julia2paddle(fn_name, setup_code):
    Main.eval(setup_code)

    class _JuliaFunction(paddle.autograd.PyLayer):
        @staticmethod
        def forward(ctx, *tensor_args):
            Main.paddle_arguments___ = tuple(x.cpu().detach().numpy() for x in tensor_args)
            y, back = Main.eval(f"Zygote.pullback({fn_name}, paddle_arguments___...)")
            ctx.back = back
            return paddle.to_tensor(y)

        @staticmethod
        # @once_differentiable
        def backward(ctx, grad_output):
            grad_inputs = ctx.back(grad_output.cpu().detach().numpy())
            grad_inputs = tuple(paddle.to_tensor(gx) for gx in grad_inputs)
            return grad_inputs

    name = f"JuliaFunction_{fn_name}"
    _JuliaFunction.__name__ = name
    _JuliaFunction.__qualname__ = name
    return _JuliaFunction.apply


class JuliaRun:
    def __init__(self):
        self.setup_code = f"""
        import DPFEHM
        import Zygote

        sidelength = 50.0#m
        thickness = 10.0#m
        mins = [-sidelength, -sidelength, 0]
        maxs = [sidelength, sidelength, thickness]
        ns = [100, 100, 1]
        meanloghyco1 = log(1e-5)#m/s
        lefthead = 1.0#m
        righthead = 0.0#m
        coords, neighbors, areasoverlengths, _ = DPFEHM.regulargrid2d(mins, maxs, ns, 1.0)
        Qs = zeros(size(coords, 2))

        logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] 
                                      .+ Ks[map(p->p[2], neighbors)]))

        boundaryhead(x, y) =  (x - maxs[1]) / (mins[1] - maxs[1]) + 1           
        dirichletnodes = Int[]
        dirichleths = zeros(size(coords, 2))
        for i = 1:size(coords, 2)
                if coords[1, i] == mins[1] || coords[1, i] == maxs[1]
                        push!(dirichletnodes, i)
                        dirichleths[i] = boundaryhead(coords[1:2, i]...)
                end
        end

        function solveforp(logKs)
                @assert length(logKs) == length(Qs)
                Ks_neighbors = logKs2Ks_neighbors(logKs)
                return DPFEHM.groundwater_steadystate(Ks_neighbors, neighbors, 
                                                      areasoverlengths, dirichletnodes, 
                                                      dirichleths, Qs)
        end
        """

        self.func_name = "solveforp"

        self.func = julia2paddle(self.func_name, self.setup_code)
