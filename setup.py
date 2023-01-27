from setuptools import setup, find_packages

setup(
    name="planetsynth",
    version="1.0.2",
    description="a python tool to generate cooling tracks for giant planets",
    url="https://github.com/tiny-hippo/planetsynth",
    author="Simon Müller",
    author_email="simonandres.mueller@uzh.ch",
    license="MIT",
    packages=find_packages(include=["planetsynth", "planetsynth.*"]),
    package_data={"planetsynth": ["interpolators/*.zip*"]},
    python_requires=">=3.8",
    install_requires=["numpy>=1.18.5", "scipy>=1.9.0"],
)
