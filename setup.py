from setuptools import setup, find_packages

setup(
    name='ocelotml',
    version='0.1',
    packages= find_packages(),
    entry_points={
        'console_scripts': [
            'ocelotml = ocelotml.main:main',
        ],
    },
)