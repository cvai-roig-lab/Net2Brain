from setuptools import setup, find_packages

setup(
    name = 'net2brain',
    version = '1.1.2',
    author = 'Roig Lab',
    packages=find_packages(),
    package_data={
        "": ["*.json"],
        "net2brain.architectures.configs": ["*.*"],
    },
    include_package_data=True,
    install_requires=[
        'flake8',
        'librosa',
        'torchlibrosa',
        'h5py',
        'eva-decord',  # for video models on mac devices
        'sentencepiece',
        'flax',
        'transformers',
        'einops',
        'accelerate',
        'matplotlib',
        'statsmodels',
        'requests',
        'seaborn==0.12.2',
        'opencv_python_headless',
        'pandas',
        'numpy',
        'Pillow',
        'prettytable',
        'gdown',
        'pycocotools',
        'pytest',
        'scikit_learn',
        'scipy',
        'torch',
        'tqdm',
        'visualpriors == 0.3.5',
        'timm == 0.4.12',
        'torchextractor == 0.3.0',
        'torchvision',
        'rsatoolbox == 0.0.3',
        'pytorchvideo @ git+https://github.com/facebookresearch/pytorchvideo.git@eb04d1b', # NOTE: Change when new release published
        'clip @ git+https://github.com/openai/CLIP.git',
        'mit_semseg @ git+https://github.com/CSAILVision/semantic-segmentation-pytorch.git@master'
    ]
)
