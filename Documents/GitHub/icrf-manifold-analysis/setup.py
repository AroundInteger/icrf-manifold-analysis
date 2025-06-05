from setuptools import setup, find_packages

setup(
    name="icrf-manifold-analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "fcsparser>=0.2.0",
        "jupyter>=1.0.0",
        "notebook>=6.4.0",
    ],
    author="AroundInteger",
    author_email="your.email@example.com",
    description="Analysis of cell cycle dynamics using manifold learning techniques",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AroundInteger/icrf-manifold-analysis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 