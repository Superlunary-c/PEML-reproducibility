"""Dataset builders for CIFAR-100, ImageNet-100, and CIFAR100-LT.

Expected config fields
----------------------
Common fields (required unless a default is stated below):

- cfg.DATASET.PATH.ROOT : str or path-like
    Dataset root.
    * CIFAR-100: download/cache directory.
    * ImageNet-100: directory containing `train/` and `val/` subdirectories.
- cfg.DATASET.Mean : sequence of 3 floats (optional)
- cfg.DATASET.Std  : sequence of 3 floats (optional)
- cfg.MODEL.IMG_SIZE : int (optional)

Additional CIFAR100-LT fields (optional; defaults shown):

- cfg.DATASET.SEED      (default: 0)
- cfg.DATASET.IR        (default: 100)
- cfg.DATASET.N_MAX     (default: 500)
- cfg.DATASET.MIN_COUNT (default: 1)

Defaults
--------
If not provided, dataset-specific defaults are used:

- CIFAR-100 / CIFAR100-LT:
    Mean=(0.5071, 0.4867, 0.4408)
    Std=(0.2675, 0.2565, 0.2761)
    IMG_SIZE=32
- ImageNet-100:
    Mean=(0.485, 0.456, 0.406)
    Std=(0.229, 0.224, 0.225)
    IMG_SIZE=224

ImageNet-100 directory layout
-----------------------------
The ImageNet-100 loader expects:

ROOT/
  train/
    n01440764/
    n01443537/
    ...
  val/
    n01440764/
    n01443537/
    ...

where the class-folder names are ImageNet synset IDs. Only the 100 synsets
listed in `IMAGENET100_CLASSES` are used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import numbers

import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

__all__ = [
    "IMAGENET100_CLASSES",
    "ImageNet100",
    "TwoCropTransform",
    "get_cifar100_dataset",
    "get_imagenet100_dataset",
    "get_cifar100_lt_dataset",
]


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_MISSING = object()


# From https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
IMAGENET100_CLASSES = (
    'n02869837', 'n01749939', 'n02488291', 'n02107142', 'n13037406',
    'n02091831', 'n04517823', 'n04589890', 'n03062245', 'n01773797',
    'n01735189', 'n07831146', 'n07753275', 'n03085013', 'n04485082',
    'n02105505', 'n01983481', 'n02788148', 'n03530642', 'n04435653',
    'n02086910', 'n02859443', 'n13040303', 'n03594734', 'n02085620',
    'n02099849', 'n01558993', 'n04493381', 'n02109047', 'n04111531',
    'n02877765', 'n04429376', 'n02009229', 'n01978455', 'n02106550',
    'n01820546', 'n01692333', 'n07714571', 'n02974003', 'n02114855',
    'n03785016', 'n03764736', 'n03775546', 'n02087046', 'n07836838',
    'n04099969', 'n04592741', 'n03891251', 'n02701002', 'n03379051',
    'n02259212', 'n07715103', 'n03947888', 'n04026417', 'n02326432',
    'n03637318', 'n01980166', 'n02113799', 'n02086240', 'n03903868',
    'n02483362', 'n04127249', 'n02089973', 'n03017168', 'n02093428',
    'n02804414', 'n02396427', 'n04418357', 'n02172182', 'n01729322',
    'n02113978', 'n03787032', 'n02089867', 'n02119022', 'n03777754',
    'n04238763', 'n02231487', 'n03032252', 'n02138441', 'n02104029',
    'n03837869', 'n03494278', 'n04136333', 'n03794056', 'n03492542',
    'n02018207', 'n04067472', 'n03930630', 'n03584829', 'n02123045',
    'n04229816', 'n02100583', 'n03642806', 'n04336792', 'n03259280',
    'n02116738', 'n02108089', 'n03424325', 'n01855672', 'n02090622',
)


class ImageNet100(ImageFolder):
    """ImageFolder variant that keeps only the predefined ImageNet-100 synsets.

    The loader expects an ImageFolder-style directory where each subdirectory is
    a synset folder name (for example, ``n02123045``).
    """

    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(
                f"ImageNet100 expected a directory at '{root}', but it does not exist."
            )

        allowed = set(IMAGENET100_CLASSES)
        classes = sorted(
            entry.name for entry in root.iterdir() if entry.is_dir() and entry.name in allowed
        )

        missing = sorted(allowed.difference(classes))
        if missing:
            preview = ", ".join(missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise FileNotFoundError(
                f"ImageNet100 expected {len(IMAGENET100_CLASSES)} synset folders under '{root}', "
                f"but {len(missing)} are missing. Missing examples: {preview}{suffix}"
            )

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class TwoCropTransform:
    """Create two stochastic views of the same image.

    This wrapper is commonly used for contrastive learning, where each dataset
    item should return two independently augmented versions of the same source
    image.
    """

    def __init__(self, transform: transforms.Compose):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def _get_cfg_value(cfg: Any, path: str, default: Any = _MISSING) -> Any:
    """Read a dotted config path from an attribute-style object or nested dict.

    Parameters
    ----------
    cfg:
        Config object. Nested dictionaries and attribute-style configs are both
        supported.
    path:
        Dotted path such as ``DATASET.PATH.ROOT``.
    default:
        Optional default to return when the path is missing.

    Raises
    ------
    ValueError
        If the path is missing and no default is provided.
    """
    current = cfg
    for part in path.split('.'):
        try:
            if isinstance(current, Mapping):
                current = current[part]
            else:
                current = getattr(current, part)
        except (KeyError, AttributeError, TypeError):
            if default is not _MISSING:
                return default
            raise ValueError(
                f"Missing required config field `{path}`. "
                f"Please define it in your config before calling this dataset builder."
            ) from None

    if current is None and default is not _MISSING:
        return default
    if current is None and default is _MISSING:
        raise ValueError(f"Config field `{path}` is set to None, but a value is required.")
    return current


def _as_triplet(name: str, value: Any) -> tuple[float, float, float]:
    """Validate and convert a length-3 numeric sequence to a float tuple."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"`{name}` must be a sequence of 3 numeric values, got: {value!r}")

    out = []
    for x in value:
        if not isinstance(x, numbers.Real):
            raise ValueError(f"`{name}` must contain numeric values, got: {value!r}")
        out.append(float(x))
    return tuple(out)  # type: ignore[return-value]


def _as_positive_int(name: str, value: Any) -> int:
    """Validate and convert a positive integer config value."""
    if not isinstance(value, numbers.Integral) or int(value) <= 0:
        raise ValueError(f"`{name}` must be a positive integer, got: {value!r}")
    return int(value)


def _as_nonnegative_int(name: str, value: Any) -> int:
    """Validate and convert a nonnegative integer config value."""
    if not isinstance(value, numbers.Integral) or int(value) < 0:
        raise ValueError(f"`{name}` must be a nonnegative integer, got: {value!r}")
    return int(value)


def _validate_views(views: int) -> int:
    """Ensure ``views`` is either 1 or 2."""
    if views not in (1, 2):
        raise ValueError(f"`views` must be 1 or 2, got {views!r}")
    return int(views)


def _resolve_root(cfg: Any, for_imagenet: bool) -> Path:
    """Resolve and validate the dataset root path.

    CIFAR-100 downloads can create the directory if it does not exist.
    ImageNet-100 requires an existing root containing ``train/`` and ``val/``.
    """
    root = Path(_get_cfg_value(cfg, "DATASET.PATH.ROOT"))

    if for_imagenet:
        train_root = root / "train"
        val_root = root / "val"
        if not train_root.is_dir() or not val_root.is_dir():
            raise FileNotFoundError(
                "ImageNet100 expects `cfg.DATASET.PATH.ROOT` to contain `train/` and `val/` "
                f"subdirectories, but got root='{root}'."
            )
    else:
        root.mkdir(parents=True, exist_ok=True)

    return root


def _resolve_cifar_cfg(cfg: Any) -> tuple[Path, tuple[float, float, float], tuple[float, float, float], int]:
    """Resolve CIFAR-100 dataset settings with sensible defaults."""
    root = _resolve_root(cfg, for_imagenet=False)
    mean = _as_triplet("cfg.DATASET.Mean", _get_cfg_value(cfg, "DATASET.Mean", CIFAR100_MEAN))
    std = _as_triplet("cfg.DATASET.Std", _get_cfg_value(cfg, "DATASET.Std", CIFAR100_STD))
    img_size = _as_positive_int("cfg.MODEL.IMG_SIZE", _get_cfg_value(cfg, "MODEL.IMG_SIZE", 32))
    return root, mean, std, img_size


def _resolve_imagenet_cfg(cfg: Any) -> tuple[Path, tuple[float, float, float], tuple[float, float, float], int]:
    """Resolve ImageNet-100 dataset settings with sensible defaults."""
    root = _resolve_root(cfg, for_imagenet=True)
    mean = _as_triplet("cfg.DATASET.Mean", _get_cfg_value(cfg, "DATASET.Mean", IMAGENET_MEAN))
    std = _as_triplet("cfg.DATASET.Std", _get_cfg_value(cfg, "DATASET.Std", IMAGENET_STD))
    img_size = _as_positive_int("cfg.MODEL.IMG_SIZE", _get_cfg_value(cfg, "MODEL.IMG_SIZE", 224))
    return root, mean, std, img_size


def _resolve_lt_cfg(cfg: Any) -> tuple[int, int, int, int]:
    """Resolve long-tail generation settings for CIFAR100-LT."""
    seed = _as_nonnegative_int("cfg.DATASET.SEED", _get_cfg_value(cfg, "DATASET.SEED", 0))
    ir = _as_positive_int("cfg.DATASET.IR", _get_cfg_value(cfg, "DATASET.IR", 100))
    n_max = _as_positive_int("cfg.DATASET.N_MAX", _get_cfg_value(cfg, "DATASET.N_MAX", 500))
    min_count = _as_positive_int(
        "cfg.DATASET.MIN_COUNT", _get_cfg_value(cfg, "DATASET.MIN_COUNT", 1)
    )
    return seed, ir, n_max, min_count


def _maybe_resize_for_eval(img_size: int, default_size: int) -> list[Any]:
    """Add a resize op for evaluation only when the requested size differs.

    This keeps the original behavior for the common case (e.g., CIFAR at 32x32)
    while avoiding shape mismatches when users override `IMG_SIZE`.
    """
    if img_size == default_size:
        return []
    return [transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC)]


def _build_cifar_train_transform(mean, std, img_size: int) -> transforms.Compose:
    """Build the CIFAR-100 training augmentation pipeline."""
    normalize = transforms.Normalize(mean=mean, std=std)
    return transforms.Compose([
        transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])


def _build_cifar_eval_transform(mean, std, img_size: int) -> transforms.Compose:
    """Build the CIFAR-100 evaluation transform."""
    normalize = transforms.Normalize(mean=mean, std=std)
    return transforms.Compose([
        *_maybe_resize_for_eval(img_size, default_size=32),
        transforms.ToTensor(),
        normalize,
    ])


def _build_imagenet_train_transform(mean, std, img_size: int) -> transforms.Compose:
    """Build the ImageNet-100 training augmentation pipeline."""
    normalize = transforms.Normalize(mean=mean, std=std)
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=img_size,
            scale=(0.08, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])


def _build_imagenet_eval_transform(mean, std, img_size: int) -> transforms.Compose:
    """Build the ImageNet-100 evaluation transform."""
    normalize = transforms.Normalize(mean=mean, std=std)
    resize_size = int(round(img_size * 256 / 224))
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])


def _wrap_for_views(train_transform: transforms.Compose, views: int):
    """Return either a single-view transform or a two-view wrapper."""
    views = _validate_views(views)
    return train_transform if views == 1 else TwoCropTransform(train_transform)


def get_cifar100_dataset(cfg: Any, views: int = 2):
    """Build balanced CIFAR-100 train/validation datasets.

    Parameters
    ----------
    cfg:
        Config object or nested dict.
    views:
        1 for standard single-view training, 2 for contrastive two-view training.

    Returns
    -------
    train_dataset, val_dataset
    """
    root, mean, std, img_size = _resolve_cifar_cfg(cfg)
    train_transform = _build_cifar_train_transform(mean, std, img_size)
    valid_transform = _build_cifar_eval_transform(mean, std, img_size)

    train_dataset = datasets.CIFAR100(
        root=str(root),
        train=True,
        transform=_wrap_for_views(train_transform, views),
        download=True,
    )
    val_dataset = datasets.CIFAR100(
        root=str(root),
        train=False,
        transform=valid_transform,
        download=True,
    )
    return train_dataset, val_dataset


def get_imagenet100_dataset(cfg: Any, views: int = 2):
    """Build ImageNet-100 train/validation datasets.

    Notes
    -----
    `cfg.DATASET.PATH.ROOT` must point to a directory containing `train/` and
    `val/` subdirectories. Only synsets in `IMAGENET100_CLASSES` are loaded.
    """
    root, mean, std, img_size = _resolve_imagenet_cfg(cfg)
    train_transform = _build_imagenet_train_transform(mean, std, img_size)
    valid_transform = _build_imagenet_eval_transform(mean, std, img_size)

    train_dataset = ImageNet100(
        root=str(root / "train"),
        transform=_wrap_for_views(train_transform, views),
    )
    val_dataset = ImageNet100(
        root=str(root / "val"),
        transform=valid_transform,
    )
    return train_dataset, val_dataset


def _long_tail_counts(C: int = 100, n_max: int = 500, IR: int = 100, min_count: int = 1) -> np.ndarray:
    """Compute power-law class counts for a long-tailed dataset.

    The counts follow:
        n_j = n_max * IR^(-(j)/(C-1)) for j=0..C-1

    Parameters
    ----------
    C:
        Number of classes.
    n_max:
        Number of samples for the head class.
    IR:
        Imbalance ratio (head:tail = IR:1).
    min_count:
        Minimum number of samples allowed for any class.
    """
    if C <= 0:
        raise ValueError(f"`C` must be positive, got {C}")
    if n_max <= 0:
        raise ValueError(f"`n_max` must be positive, got {n_max}")
    if IR <= 0:
        raise ValueError(f"`IR` must be positive, got {IR}")
    if min_count <= 0:
        raise ValueError(f"`min_count` must be positive, got {min_count}")

    j = np.arange(C)
    ratios = IR ** (-(j / max(C - 1, 1)))
    raw = n_max * ratios
    counts = np.maximum(min_count, np.round(raw).astype(int))
    return counts


def _make_cifar100_lt_indices(
    targets,
    IR: int = 100,
    n_max: int = 500,
    min_count: int = 1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample CIFAR-100 training indices to create a long-tailed split.

    Parameters
    ----------
    targets:
        Class labels for the full CIFAR-100 training set.
    IR:
        Imbalance ratio.
    n_max:
        Head-class count.
    min_count:
        Minimum class count after subsampling.
    seed:
        Random seed controlling the class-wise subsampling.
    """
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    if targets.ndim != 1:
        raise ValueError(f"`targets` must be a 1D array-like object, got shape {targets.shape}")

    C = 100
    counts = _long_tail_counts(C=C, n_max=n_max, IR=IR, min_count=min_count)

    all_indices = []
    for cls in range(C):
        cls_idx = np.where(targets == cls)[0]
        if len(cls_idx) == 0:
            raise ValueError(f"Class {cls} has no samples in the provided targets.")
        rng.shuffle(cls_idx)
        n_keep = min(int(counts[cls]), len(cls_idx))
        all_indices.append(cls_idx[:n_keep])

    all_indices = np.concatenate(all_indices)
    rng.shuffle(all_indices)
    return all_indices, counts


def get_cifar100_lt_dataset(cfg: Any, views: int = 2):
    """Build CIFAR100-LT (imbalanced training set + balanced test set).

    Returns
    -------
    train_dataset_lt, val_dataset, counts
        `counts` is a NumPy array of length 100 containing the per-class train
        counts used to create the long-tailed split.
    """
    root, mean, std, img_size = _resolve_cifar_cfg(cfg)
    seed, ir, n_max, min_count = _resolve_lt_cfg(cfg)

    train_transform = _build_cifar_train_transform(mean, std, img_size)
    valid_transform = _build_cifar_eval_transform(mean, std, img_size)

    # Load the raw training labels once to compute the long-tail subset indices.
    base_train = datasets.CIFAR100(
        root=str(root),
        train=True,
        transform=None,
        download=True,
    )
    indices, counts = _make_cifar100_lt_indices(
        targets=base_train.targets,
        IR=ir,
        n_max=n_max,
        min_count=min_count,
        seed=seed,
    )

    longtail_train_full = datasets.CIFAR100(
        root=str(root),
        train=True,
        transform=_wrap_for_views(train_transform, views),
        download=False,
    )
    train_dataset_lt = Subset(longtail_train_full, indices.tolist())

    val_dataset = datasets.CIFAR100(
        root=str(root),
        train=False,
        transform=valid_transform,
        download=True,
    )
    return train_dataset_lt, val_dataset, counts
