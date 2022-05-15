# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='simple_ml_model',
    version='1.0',
    description='Simple ML model demo',
    author='Minghao Liu',
    author_email='minghaoliu1993@gmail.com',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'scikit-learn',
        'pandas',
    ],
    url='https://github.com/MHLIU93/simple_ml_model',
    entry_points={
        'console_scripts': [
            'simple_ml_model = simple_ml_model.main:train_model',
        ],
    },
    zip_safe=False,
)
