from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np
import os
import platform
import sys

# Detect compiler options based on platform
compiler_args = []
link_args = []

if platform.system() == 'Windows':
    # For Windows with MSVC compiler
    compiler_args.extend([
        '/O2',  # Optimize for speed
        '/fp:fast',  # Fast floating point calculations
        '/Gy',  # Enable function-level linking
        '/Oi',  # Enable intrinsics
        '/GL',  # Whole program optimization
    ])
    link_args.extend([
        '/LTCG',  # Link-time code generation
    ])
else:
    # For GCC/Clang
    compiler_args.extend([
        '-O3',  # Optimize for speed
        '-march=native',  # Use native CPU architecture features
        '-mtune=native',  # Tune for native CPU
        '-ffast-math',  # Fast math operations
        '-funroll-loops',  # Unroll loops for performance
        '-ftree-vectorize',  # Enable vectorization
    ])
    
    # Remove hardcoded SIMD instruction sets - let the compiler decide based on -march=native
    # We'll detect features at runtime instead
    # if platform.machine() in ('x86_64', 'amd64', 'AMD64'):
    #     compiler_args.extend([
    #         '-mavx2',  # AVX2 instructions
    #         '-mfma',  # FMA instructions
    #         '-msse4.1',  # SSE4.1 instructions
    #     ])

# Define Cython extensions
extensions = [
    Extension(
        "fast_operations",
        ["fast_operations.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compiler_args,
        extra_link_args=link_args,
        language="c++"
    ),
    Extension(
        "simd_operations",
        ["simd_operations.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compiler_args,
        extra_link_args=link_args,
        language="c++"
    )
]

# Setup configuration
setup(
    name="waste_detection",
    version="2.0.0",
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': True,
            'wraparound': True,
            'initializedcheck': True,
            'cdivision': True,
            'infer_types': True,
        },
        annotate=True  # Generate HTML annotation of Python->C conversions
    ),
    include_dirs=[np.get_include()],
    install_requires=[
        "numpy>=1.22.0",
        "opencv-python-headless>=4.7.0",
        "psutil>=5.9.0",
        "Cython>=0.29.33",
        "pillow>=9.0.0",
        "imagehash>=4.3.1",
        "matplotlib>=3.7.0",
        "pywin32>=301; platform_system=='Windows'",
        "pygrabber>=0.1; platform_system=='Windows'",
        "pycuda>=2022.2; platform_system=='Windows'",
        "tensorflow>=2.11.0",
        "onnxruntime-gpu>=1.14.0; platform_system=='Windows'",
        "onnxruntime>=1.14.0; platform_system!='Windows'",
        "scikit-learn>=1.2.0",
    ],
    python_requires=">=3.8",
    author="Waste Detection Team",
    description="Advanced waste detection system with optimized performance",
) 