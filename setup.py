from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ECE277_WI23',
    packages=find_packages(),
    version='1.0.0',
    author='LingYi Li & HuaYue Li',
    ext_modules=[
        CUDAExtension(
            'sharpening_sum', # operator name
            ['./ops/src/Laplace_Sharpening/sharpening.cpp',
             './ops/src/Laplace_Sharpening/sharpening_cuda.cu',]
        ),
        CUDAExtension(
            'sharpening_add', # operator name
            ['./ops/src/Image_Add/add_image.cpp',
             './ops/src/Image_Add/add_image_cuda.cu',]
        ),
        CUDAExtension(
            'dct',  # operator name
            ['./ops/src/DCT/dct_mul.cpp',
             './ops/src/DCT/dct_cuda.cu', ]
        ),
        CUDAExtension(
            'dct2', # operator name
            ['./ops/src/DCT_2/dct_2_mul.cpp',
             './ops/src/DCT_2/dct_2_cuda.cu',]
        ),
        CUDAExtension(
            'comp', # operator name
            ['./ops/src/Compress/comp.cpp',
             './ops/src/Compress/comp_cuda.cu',]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)