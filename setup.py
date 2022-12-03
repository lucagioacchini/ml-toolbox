import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='mltoolbox',
    packages=['mltoolbox'],
    install_requires=['requests']
)
