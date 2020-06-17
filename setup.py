import sys
from setuptools import setup, find_packages


if sys.version_info < (3, 6):
    sys.stderr.write(
        f'You are using Python '
        + "{'.'.join(str(v) for v in sys.version_info[:3])}.\n\n"
        + 'aydin only supports Python 3.6 and above.\n\n'
        + 'Please install Python 3.6 using:\n'
        + '  $ pip install python==3.6\n\n'
    )
    sys.exit(1)

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ssi',
    version='0.0.1',
    install_requires=required,
    packages=find_packages(),
)
