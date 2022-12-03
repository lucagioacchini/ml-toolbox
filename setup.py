import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='mltoolbox',
    version='0.0.1',
    author='Luca Gioacchini',
    author_email='luca.gioacchini@polito.it',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/lucagioacchini/MLToolbox',
    packages=['mltoolbox']
)
