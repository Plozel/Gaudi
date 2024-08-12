from setuptools import setup, find_packages

setup(
    name='gaudi',
    version='0.1.0',
    author='Oren Ploznik',
    author_email='ploznik@campus.technion.ac.il',
    description='Gaudi is a Python package designed for analyzing spatially resolved transcriptomics data, offering unique insights through a community-level perspective. It facilitates comprehensive data integration and visualization to support advanced biological research.',
    packages=find_packages(include=['gaudi', 'gaudi.*']),
    package_data={
        'gaudi': ['assets/cellchatdb/*.csv'],
    },
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.10',
)
