from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='StellarPolAnalyzer',
    version='0.1.19',
    description='Librería Python que automatiza un completo un pipeline polarimétrico sobre imágenes FITS astronómicas: Empareja, fotometriza y astrometriza imágenes polarimétricas de campo estelar en un solo flujo.',
    author='Oscar Mellizo Angulo',
    author_email='omellizo@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/oscarmellizo/StellarPolAnalyzer',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'astropy',
        'photutils',
        'scikit-learn',
        'scikit-image',
        'scipy',
        'astroquery',
        'reportlab'
    ],
    license='Apache License 2.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
