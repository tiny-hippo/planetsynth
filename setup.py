from setuptools import setup, find_packages

setup(
    name="planetsynth",
    version="1.0.1",
    description="a python tool to generate cooling tracks for giant planets",
    url="",
    author="Simon Müller",
    author_email="simon.mueller7@uzh.ch",
    license="MIT",
    packages=find_packages(include=["planetsynth", "planetsynth.*"]),
    package_data={"planetsynth": ["interpolators/*.zip*"]},
    install_requires=["numpy", "scipy"],
)
