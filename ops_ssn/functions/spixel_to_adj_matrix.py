import torch

import spixel_to_adj_matrix_cuda


class SPixel2DToAdjMatrixFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, input, n_pixel):
        return spixel_to_adj_matrix_cuda.spixel2d_forward(input, n_pixel)

    @staticmethod
    def backward(self, grad):
        raise NotImplementedError('This function do not support backward')


class SPixel3DToAdjMatrixFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, input, n_pixel):
        return spixel_to_adj_matrix_cuda.spixel3d_forward(input, n_pixel)

    @staticmethod
    def backward(self, grad):
        raise NotImplementedError('This function do not support backward')
