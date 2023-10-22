from setuptools import setup, find_packages

setup(
    name='dune_imperium',
    version='0.1.0',
    description='Remote Dune Imperium Gym Environment',
    packages=find_packages(),
    install_requires=[
        'gym>=0.9.4,<=0.15.7',
        'numpy>=1.13.0',
        'opencv-python>=3.4.2.0',
        'requests',
        'flask'
    ]
)
