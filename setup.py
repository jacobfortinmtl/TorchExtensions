from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='nan_cpp',
    ext_modules=[cpp_extension.CppExtension('nan_cpp', ['nan.cpp'], extra_compile_args=["-g"])],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
