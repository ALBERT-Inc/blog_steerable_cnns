from steerable_cnns import __version__
from setuptools import find_packages, setup


def get_requirements():
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    return requirements


def get_developer_requirements():
    with open('requirements-dev.txt') as f:
        requirements = f.read().splitlines()
    return requirements


setup(
    name='steerable-cnns',
    version=__version__,
    author='Shunsuke Shimizu',
    author_email='shunsuke_shimizu@albert2005.co.jp',
    packages=find_packages(),
    install_requires=get_requirements(),
    extras_require={
        'dev': get_developer_requirements(),
    },
)
