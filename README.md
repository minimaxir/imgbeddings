# imgbeddings

A Python package to generate embedding vectors from images, using OpenAI's robust CLIP model via HuggingFace Transformers. These image embeddings can be used for many things: unsupervised clustering (e.g. via umap), embeddings search (e.g. via faiss), and using downstream for other ML/AI tasks.

## Ethics

## Notes

- This package only works with image data intentionally as opposed to leveraging CLIP's ability to link image and text. For downstream tasks, using your own text will likely give bette results.

## License

MIT
