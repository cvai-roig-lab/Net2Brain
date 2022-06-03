from setuptools import setup

setup(
    name = 'net2brain',
    version = '0.1.0',
    author = 'Roig Lab',
    packages=['net2brain'],
    install_requires=[
        'matplotlib',
        'numpy',
        'opencv_python_headless',
        'pandas',
        'prettytable',
        'scikit_learn',
        'scipy',
        'Pillow',
        'tqdm',
        'PyQt5',
        'h5py',
        'pytest',
        'visualpriors == 0.3.5',
        'torch == 1.10.2',
        'torchvision == 0.11.3',
        'timm == 0.4.12',
        'torchextractor == 0.3.0',
        'pytorchvideo == 0.1.5',
        'rsatoolbox == 0.0.3',
        'cornet @ git+https://github.com/dicarlolab/CORnet'
    ]
)
