from setuptools import setup

setup(
    name="imgbeddings",
    packages=["imgbeddings"],  # this must be the same as the name above
    version="0.1.0",
    description="A Python package to generate image embeddings with CLIP without PyTorch/TensorFlow",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Max Woolf",
    author_email="max@minimaxir.com",
    url="https://github.com/minimaxir/imgbeddings",
    keywords=[
        "ai",
        "transformers",
        "onnx",
        "images",
        "image-processing",
        "embeddings",
        "clip",
    ],
    classifiers=[],
    license="MIT",
    python_requires=">=3.7",
    include_package_data=True,
    install_requires=[
        "transformers>=4.17.0",
        "onnxruntime>=1.10.0",
        "Pillow",
        "tqdm",
        "scikit-learn",
    ],
)
