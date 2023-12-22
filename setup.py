from setuptools import setup, find_packages

setup(
    name='cve',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv",
        "matplotlib"
    ],
    author='Sam Pearson-Smith',
    author_email='samps1@outlook.com',
    description='Extension of OpenCV4',
    long_description='Adds more functionality, input validation for safer usage, extended with more features',
    long_description_content_type='text/markdown',
    url='https://github.com/SamP-S/OpenCV_Extended',  # URL to the package repository
    license='GPL-3.0',  # Specify the license type
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GPL Liscense',
        'Operating System :: OS Independent',
        'OpenCV :: Computer Vision'
    ],
)