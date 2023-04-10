from torchvision.transforms import (
    ToTensor,
    Compose,
    Normalize,
    Resize,
    RandomHorizontalFlip,
    Grayscale,
)
from torchvision.transforms import (
    RandomResizedCrop,
    RandomHorizontalFlip,
)


def build(config):
    aug_transform = None

    if config.mode == "pretrain":
        transform = Compose(
            [Grayscale(), Resize((config.frame_size, config.frame_size), antialias=True)]
        )
    else:
        transform = Compose(
            [
                ToTensor(),
                Resize((config.frame_size, config.frame_size), antialias=True),
                Normalize((config.mean), (config.std)),
            ]
        )

    # Make sure the new transforms are working as they should
    aug_transform = Compose(
        [
            RandomResizedCrop(size=config.frame_size, scale=(0.7, 1), antialias=True),
            RandomHorizontalFlip(p=0.3),
        ]
    )

    return transform, aug_transform
