from setuptools import setup, find_packages

setup(
    name="zahramun",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29",
        "stable-baselines3>=2.0",
        "sb3-contrib>=2.0",
        "blessed>=1.20",
        "numpy>=1.24",
        "scipy>=1.10",
        "PyYAML>=6.0",
        "matplotlib>=3.8",
    ],
    python_requires=">=3.11",
)
