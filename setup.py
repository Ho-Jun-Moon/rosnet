import setuptools

setuptools.setup(
    name="rosnet", 
    version="0.0.1",
    author="papamoon0113",
    author_email="papamoon0113@pusan.ac.kr",
    description="Causal discovery with ML",
    long_description=open('README.md', encoding='UTF-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ho-Jun-Moon/rosnet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = ['numpy'],
)   