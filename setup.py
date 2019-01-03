from setup import find_packages, setup

REQUIRED_PACKAGES = []

setup(
    name='unet',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description=''
)
