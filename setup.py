from setuptools import setup, find_packages


def readme():
  with open('README_lib.md', 'r', encoding='utf-8') as f:
    return f.read()


setup(
  name='dnn_tts_torch',
  version='0.0.9',
  author='KirillPogranichny',
  author_email='kirillpogranichny@gmail.com',
  description='This is a library consisting of pre-trained models for the synthesis of Russian and English speech',
  long_description=readme(),
  long_description_content_type='text/markdown',
  url='https://github.com/KirillPogranichny/dnn_tts_torch.git',
  packages=find_packages(include=['dnn_tts_torch', 'dnn_tts_torch.*']),
  install_requires=[
    'absl-py==2.1.0',
    'audioread==3.0.1',
    'beautifulsoup4==4.12.3',
    'certifi==2024.6.2',
    'cffi==1.16.0',
    'charset-normalizer==3.3.2',
    'colorama==0.4.6',
    'decorator==5.1.1',
    'filelock==3.15.3',
    'fsspec==2024.6.0',
    'gdown==5.2.0',
    'grpcio==1.64.1',
    'idna==3.7',
    'imageio==2.34.1',
    'intel-openmp==2021.4.0',
    'Jinja2==3.1.4',
    'joblib==1.4.2',
    'lazy_loader==0.4',
    'librosa==0.10.2.post1',
    'llvmlite==0.43.0',
    'Markdown==3.6',
    'MarkupSafe==2.1.5',
    'mkl==2021.4.0',
    'mpmath==1.3.0',
    'msgpack==1.0.8',
    'networkx==3.3',
    'numba==0.60.0',
    'numpy==1.26.4',
    'packaging==24.1',
    'pillow==10.3.0',
    'platformdirs==4.2.2',
    'pooch==1.8.2',
    'protobuf==4.25.3',
    'pycparser==2.22',
    'PySocks==1.7.1',
    'requests==2.32.3',
    'scikit-image==0.24.0',
    'scikit-learn==1.5.0',
    'scipy==1.13.1',
    'setuptools==70.1.0',
    'six==1.16.0',
    'soundfile==0.12.1',
    'soupsieve==2.5',
    'soxr==0.3.7',
    'sympy==1.12.1',
    'tbb==2021.13.0',
    'tensorboard==2.17.0',
    'tensorboard-data-server==0.7.2',
    'tensorboardX==2.6.2.2',
    'threadpoolctl==3.5.0',
    'tifffile==2024.6.18',
    'torch==2.3.1',
    'tqdm==4.66.4',
    'typing_extensions==4.12.2',
    'urllib3==2.2.2',
    'Werkzeug==3.0.3'
  ],
  classifiers=[
    'Programming Language :: Python :: 3.12',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent'
  ],
  keywords='tts speech synthesis',
  project_urls={
    'GitHub': 'https://github.com/KirillPogranichny/dnn_tts_torch.git'
  },
  python_requires='>=3.12'
)
