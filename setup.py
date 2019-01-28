from setuptools import find_packages, setup

REQUIRED_PACKAGES = []

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='unet',
    version='0.1',
    description='Semantic segmentation model',
    long_description=long_description,
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
)
