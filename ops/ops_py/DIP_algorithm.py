import torch
from torch.autograd import Function
import sharpening_sum
import sharpening_add
import dct
import dct2
import comp

class Sharpening(Function):

    @staticmethod
    def forward(ctx, array1, array2):
        array1 = array1.float()
        array2 = array2.float()
        row = array1.shape[0]
        col = array1.shape[1]
        ans = array1.new_zeros((row-2, col-2))
        sharpening_sum.forward(array1.contiguous(), array2.contiguous(), ans)

        # ctx.mark_non_differentiable(ans) # if the function is no need for backpropogation
        return ans

    @staticmethod
    def backward(ctx, g_out):
        # return None, None   # if the function is no need for backpropogation

        g_in1 = g_out.clone()
        g_in2 = g_out.clone()
        return g_in1, g_in2

class Sharpening_Add(Function):

    @staticmethod
    def forward(ctx, array1, array2):
        array1 = array1.float()
        array2 = array2.float()
        ans = array1.new_zeros(array1.shape)
        sharpening_add.forward(array1.contiguous(), array2.contiguous(), ans)

        # ctx.mark_non_differentiable(ans) # if the function is no need for backpropogation

        return ans

    @staticmethod
    def backward(ctx, g_out):
        # return None, None   # if the function is no need for backpropogation
        g_in1 = g_out.clone()
        g_in2 = g_out.clone()
        return g_in1, g_in2

class Dct(Function):

    @staticmethod
    def forward(ctx, array1, array2):
        array1 = array1.float()
        array2 = array2.float()
        ans = array1.new_zeros(array2.shape)
        dct.forward(array1.contiguous(), array2.contiguous(), ans)

        # ctx.mark_non_differentiable(ans) # if the function is no need for backpropogation

        return ans

    @staticmethod
    def backward(ctx, g_out):
        # return None, None   # if the function is no need for backpropogation
        g_in1 = g_out.clone()
        g_in2 = g_out.clone()
        return g_in1, g_in2


class Dct2(Function):

    @staticmethod
    def forward(ctx, array1, array2):
        array1 = array1.float()
        array2 = array2.float()
        ans = array1.new_zeros(array1.shape)
        dct2.forward(array1.contiguous(), array2.contiguous(), ans)

        # ctx.mark_non_differentiable(ans) # if the function is no need for backpropogation
        return ans

    @staticmethod
    def backward(ctx, g_out):
        # return None, None   # if the function is no need for backpropogation

        g_in1 = g_out.clone()
        g_in2 = g_out.clone()
        return g_in1, g_in2

class Comp(Function):

    @staticmethod
    def forward(ctx, array1, array2):
        array1 = array1.float()
        array2 = array2.float()
        ans = array1.new_zeros(array1.shape)
        comp.forward(array1.contiguous(), array2.contiguous(), ans)

        # ctx.mark_non_differentiable(ans) # if the function is no need for backpropogation
        return ans

    @staticmethod
    def backward(ctx, g_out):
        # return None, None   # if the function is no need for backpropogation

        g_in1 = g_out.clone()
        g_in2 = g_out.clone()
        return g_in1, g_in2


sharpening_op = Sharpening.apply
sharpening_add_op = Sharpening_Add.apply
dct_op = Dct.apply
dct2_op = Dct2.apply
comp_op = Comp.apply