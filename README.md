# imgbeddings

A Python package to generate embedding vectors from images, using OpenAI's robust CLIP model via HuggingFace Transformers. These image embeddings, derived from an image model that has seen the entire internet up to mid-2020, can be used for many things: unsupervised clustering (e.g. via umap), embeddings search (e.g. via faiss), and using downstream for other framework-agnostic ML/AI tasks.

Additionally, imgbeddings offers features to allow you to construct a custom principal component analysis (PCA) to reduce the dimensionality of those embeddings while maintaining similar performance, allows the embeddings to be more resilient to suboptimal image inputs, aligns the embeddings to a specific domain, and also potentially debiases some of the systemic biases from the source dataset.

## Usage

Note that CLIP was trained on square images only, and imgbeddings will pad and resize rectangular images into a square (imgbeddings deliberately does not center crop). As a result, images too wide/tall (e.g. more than a 3:1 ratio of largest dimension to smallest) will not generate robust embeddings.

## Ethics

The official paper for CLIP explicitly notes that there are inherent biases in the finished model, and that CLIP shouldn't be used in production applications as a result. My perspective is that having better tools free-and-open-source to _detect_ such issues and make it more transparent is an overall good, especially since there are less-public ways to create image embeddings that aren't as accessible. At the least, this package doesn't do anything that wasn't already available when CLIP was open-sourced in January 2021.

If you do use imgbeddings for your own project, I recommend doing a strong QA pass along a diverse set of inputs for your application, which is something you should always be doing whenever you work with machine learning, biased models or not.

imgbeddings is not responsible for malicious misuse of image embeddings.

## Design Notes

- CLIP is used due to its robust zero-shot nature and its ability to generalize to many domains even beyond what it was created for, as demonstrated with applications such as VQGAN + CLIP. Although the methodology to generate the embeddings would work with any Vision Transformer (ViT), many of the publically released pre-trained ViTs are trained on ImageNet only, which is not ideal.
- This package only works with image data intentionally as opposed to leveraging CLIP's ability to link image and text. For downstream tasks, using your own text in conjunction with an image will likely give better results. (e.g. if training a model on an image embeddings + text embeddings, feed both and let the model determine the relative importance of each for your use case)

For more miscellaneous design notes, see [DESIGN.md](DESIGN.md).

## License

MIT
