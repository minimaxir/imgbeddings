# Design

A few miscelleneous design notes for some counterintuitive product decisions with imgbeddings:

## Embeddings Derivation

- The embeddings are calculated like this:
  - Sum up the hidden states of the last _n_ Transformer layers of CLIP.
  - Average the post-softmax attention by the last _n_ Transformer layers of CLIP and across all attention heads.
  - Zero out the attention value of the first token. (it's the class token for CLIP, which is irrelevant here)
  - Reweight the other attention values such that they sum to 1
  - Multiply the summed hidden states for each position by the reweighted attention values, then sum up and return across all positions. This is effectively an attention-weighted average.
  - The technical implementation is done in `get_embeddings_from_output()` at [models.py](imgbeddings/models.py).
- Not all aspects of an image are relevant to its context. In theory, attention is an unsupervised way of identifying these relative impacts, thus an attention-weighted average can capture each patch's importance and disregard irrelevant parts of an image.
- No, there is no research on whether this approach is mathematically sound, however it's hard to argue with the results. Please feel free to petition my alma mater to revoke my Ph.D for statistical crimes, which will be difficult as I do not have a Ph.D.
  - The exporting code is open-sourced in the package itself in case there are optimizations found, and models are versioned so users can pin them.

## Model Selection

- CLIP is used due to its robust zero-shot nature and its ability to generalize to many domains even beyond what it was created for, as demonstrated with applications such as VQGAN + CLIP. Although the methodology to generate the embeddings would work with any Vision Transformer (ViT), many of the publically released pre-trained ViTs are trained on ImageNet only, which is not ideal.
  - Other more-explicit ViT models (Google's ViT, Microsoft's BeIT) were tested, but CLIP Vision had the best performance for this use case, subjectively. It may be worth it in the future to expand the export functionality to support arbitrary ViTs.

## Processing

- Square padding all input images is far superior to center cropping in this case as square pad forces the entire image to be used as input while center cropping only includes a subset.
  - In theory, attention-weighted averaging should discard the unnecessary black bars in processing.
- Black is the default color for square padding and rotation augmentation for more apples-to-apples comparisons between images of different types, plus it's possible for the source data to.
- PIL image augmentation functions are used instead of the popular torchvision package since, even if torch were still used as a dependency, the augmentations via torchvision are slightly lossy compared to PIL's augmentations which has downstream effects on embedding quality.
