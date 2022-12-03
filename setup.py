import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='MLtoolbox',
    version=None,
    author=None,
    author_email=None,
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mike-huls/toolbox',
    packages=['toolbox'],
    install_requires=['gensim', 'numpy', 'pandas', 'sklearn'],
)