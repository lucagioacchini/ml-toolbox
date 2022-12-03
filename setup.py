import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    version='0.0.1',
    name='mltoolbox',
    packages=['mltoolbox'],
    install_requires=['requests']
)
