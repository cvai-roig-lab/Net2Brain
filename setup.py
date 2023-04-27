from setuptools import setup, find_packages

setup(
    name = 'net2brain',
    version = '0.1.0',
    author = 'Roig Lab',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'flake8',
        'h5py',
        'matplotlib',
        'requests',
        'seaborn',
        'opencv_python_headless',
        'pandas==1.3.4',
        'numpy',
        'Pillow',
        'prettytable',
        'pytest',
        'pytorchvideo == 0.1.5',
        'scikit_learn',
        'scipy',
        'torch == 1.10.2',
        'tqdm',
        'visualpriors == 0.3.5',
        'timm == 0.4.12',
        'torchextractor == 0.3.0',
        'torchvision == 0.11.3',
        'rsatoolbox == 0.0.3',
        'clip @ git+https://github.com/openai/CLIP.git',
        'mit_semseg @ git+https://github.com/CSAILVision/semantic-segmentation-pytorch.git@master'
    ]
)
