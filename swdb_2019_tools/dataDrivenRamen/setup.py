from setuptools import setup, find_packages

"""
===============
dataDrivenRamen
===============

"""

setup(
    name='dataDrivenRamen',
    version='0.1.0dev',
    packages=find_packages(),
    url='',
    license='',
    author='Daril Brown, Shiva Farashahi, Emily Gelfand, Roman Levin, Courtnie Paschall',
    author_email='',
    description='Using Adaptive Locally Linear Dynamical Models to Elucidate Neural States in Passively Stimulated '
                'Rodents',
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'h5py', 'scikit-learn', 'decorator'],
    keywords=['neuroscience', 'Allen Institute', 'Dynamic Brain'],
    tests_require=['pytest', 'pytest-ordering'],
    extra_requires={}
)