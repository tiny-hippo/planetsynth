from setuptools import setup

setup(
    name="planetsynth",
    version="1.0",
    description="a python tool to generate cooling tracks for giant planets",
    url="",
    author="Simon Müller",
    author_email="simon.mueller7@uzh.ch",
    licence="MIT",
    packages=["planetsynth"],
    package_data={"planetsynth": ["interpolants/*.zip*"]},
    install_requires=["numpy", "scipy"],
)
