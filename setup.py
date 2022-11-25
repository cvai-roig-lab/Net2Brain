from setuptools import setup

setup(
    name = 'net2brain',
    version = '0.1.0',
    author = 'Roig Lab',
    packages=['net2brain'],
    install_requires=[
        'flake8',
        'h5py',
        'matplotlib',
        'numpy',
        'opencv_python_headless',
        'pandas',
        'Pillow',
        'prettytable',
        'pytest',
        'pytorchvideo == 0.1.5',
        'PyQt5',
        'scikit_learn',
        'scipy',
        'torch == 1.10.2',
        'tqdm',
        'visualpriors == 0.3.5',
        'timm == 0.4.12',
        'torchextractor == 0.3.0',
        'torchvision == 0.11.3',
        'rsatoolbox == 0.0.3',
        'cornet @ git+https://github.com/dicarlolab/CORnet'
    ]
)
