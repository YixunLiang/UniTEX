import os
import setuptools
import texturetools

with open('texturetools/readme.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="texturetools",
    version=texturetools.__version__,
    author="lightillusions",
    author_email="sxwlttsd@gmail.com",
    description="Texture Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lightillusions/TextureTools",
    packages=setuptools.find_packages(),
    package_data={
        'texturetools': [],
    },
    include_package_data=True,
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)