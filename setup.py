from setuptools import setup, find_packages

setup(
    name='mlops_project',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'jupyter',
        'pytest',
        'python-dotenv',
    ],
    author='Your Name',
    description='A sample MLOps project structure',
    url='https://github.com/yourusername/mlops_project',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)