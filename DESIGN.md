# Design

A few miscelleneous design notes for some counterintuitive product decisions with imgbeddings:

## Model Selection

- Other more-explicit ViT models (Google's ViT, Microsoft's BeIT) were tested, but CLIP Vision had the best performance for this use case, subjectively. It may be worth it in the future to expand the export functionality to support arbitrary ViTs.
