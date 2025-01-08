from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ssn',
    version='0.1',
    ext_modules=[
        CUDAExtension('pair_wise_distance_cuda', [
            'src/pair_wise_distance.cu',
        ]),
        CUDAExtension('spixel_to_adj_matrix_cuda', [
            'src/spixel_to_adj_matrix.cu'
        ])
    ],
    packages=find_packages(),
    cmdclass={'build_ext': BuildExtension})
