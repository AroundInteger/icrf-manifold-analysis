from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="icrf-manifold-analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@domain.com",
    description="Dynamic manifold learning for cell cycle perturbation studies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/icrf-manifold-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "pre-commit>=2.13.0",
        ],
        "fcs": [
            "flowkit>=0.8.0",
            "fcsparser>=0.2.0",
        ],
        "advanced": [
            "plotly>=5.0.0",
            "ipywidgets>=7.6.0",
            "pydiffmap>=0.2.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "icrf-analyze=src.cli:main",
        ],
    },
    package_data={
        "": ["*.md", "*.txt"],
        "src": ["data/examples/*"],
    },
    include_package_data=True,
    keywords="cell-cycle, flow-cytometry, manifold-learning, drug-perturbation, ICRF-193",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/icrf-manifold-analysis/issues",
        "Source": "https://github.com/yourusername/icrf-manifold-analysis",
        "Documentation": "https://icrf-manifold-analysis.readthedocs.io/",
    },
)
