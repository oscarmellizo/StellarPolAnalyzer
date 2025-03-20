from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='StellarPolAnalyzer',
    version='0.1.4',
    description='Librería para análisis de imágenes polarimétricas y detección de parejas de estrellas',
    author='Oscar Mellizo Angulo',
    author_email='omellizo@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/tu_usuario/StellarPolAnalyzer',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'astropy',
        'photutils',
        'scikit-learn',
    ],
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
)
