from setuptools import setup

setup(
    name="imgbeddings",
    packages=["imgbeddings"],  # this must be the same as the name above
    version="0.1.0",
    description="A robust Python tool for text-based AI training and generation using GPT-2.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Max Woolf",
    author_email="max@minimaxir.com",
    url="https://github.com/minimaxir/imgeddings",
    keywords=["gpt-2", "gpt2", "text generation", "ai"],
    classifiers=[],
    license="MIT",
    # entry_points={"console_scripts": ["aitextgen=aitextgen.cli:aitextgen_cli"]},
    python_requires=">=3.7",
    include_package_data=True,
    install_requires=[
        "transformers>=4.17.0",
        # "fire>=0.3.0",
        "onnxruntime>=1.10.0",
        "Pillow",
        "tqdm",
        "scikit-learn",
    ],
)
