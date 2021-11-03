import setuptools

from lightSOM.version import __version__

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="lightSOM",
    version=__version__,
    author="Vahid Moosavi and Sebastian Packmann",
    include_package_data=True,
    description="Self Organizing Maps Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['numpy >= 1.7', 'scipy >= 0.9',
                      'scikit-learn >= 0.21', 'numexpr >= 2.5'],
    url="",
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst', '*.json', '*.npy', '*.db'],
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        'Development Status :: 4 - Beta',
    ],
    python_requires='>=3.7',
)
