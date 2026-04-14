# PEML-reproducibility

Code and reproducibility materials for the Neural Networks submission on potential-based deep metric learning.

## Repository contents
- `data_build.py`: dataset construction utilities for CIFAR-100, ImageNet-100, and CIFAR100-LT.
- `imagenet100.txt`: the 100 ImageNet synset IDs used for the ImageNet-100 subset setting.

## Scope of this repository
This repository provides reproducibility materials that are authored for the paper.
It does **not** redistribute third-party raw image datasets.

## Public data sources
Users should obtain the public datasets from their official sources and then use the code in this repository to reproduce the experimental setup.

- **CIFAR-10 / CIFAR-100**: official CIFAR distribution from the University of Toronto.
- **ImageNet**: official ImageNet source, subject to its access terms.
- **ImageNet-100 setting**: defined here by the included `imagenet100.txt` file.

## Notes on redistribution
This repository does not include CIFAR-10/100 images or ImageNet images.
If you use this repository, please obtain those datasets separately from their official sources.

## Intended use
The current release is intended to support:
- dataset construction transparency,
- reproduction of the ImageNet-100 subset definition,
- reproduction of the CIFAR100-LT data setup.

## Citation
Please cite the associated paper and, once available, the Zenodo archive for this repository.
A `CITATION.cff` file is included for repository-level citation metadata.

## License
This repository is released under the MIT License. See `LICENSE`.
