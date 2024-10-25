from setuptools import setup, find_packages

setup(
    name='mltoolbox',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'scikit-learn==1.5.2',
        'keras==3.6.0',
        'tensorflow==2.18.0',
        'python-louvain==0.16',
        'pandas==2.2.3',
        'gensim==4.3.3'
    ],
    include_package_data=True,
)
