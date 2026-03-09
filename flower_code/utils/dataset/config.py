import ast

from PIL import Image
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize, ToTensor, Normalize, CenterCrop, \
    InterpolationMode


class DatasetConfig:
    # Mapping from dataset_id to (train_transforms, test_transforms)
    TRANSFORMS = {
        "uoft-cs/cifar10": (
            Compose([
                # Resize(256, interpolation=InterpolationMode.BILINEAR),
                # CenterCrop(224),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=[0.4914, 0.4822, 0.4465],
                          std=[0.2470, 0.2435, 0.2616])
            ]),
            Compose([
                # Resize(256, interpolation=InterpolationMode.BILINEAR),
                # CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.4914, 0.4822, 0.4465],
                          std=[0.2470, 0.2435, 0.2616])
            ])
        ),

        # Add other datasets here...
    }

    BATCH_KEY = {
        "uoft-cs/cifar10": "img",
        "flwrlabs/shakespeare": "x",
        "speech_commands": "data"
    }

    BATCH_VALUE = {
        "uoft-cs/cifar10": "label",
        "flwrlabs/shakespeare": "y",
        "speech_commands": "targets"
    }

    @staticmethod
    def _parse_hw_from_input_shape(input_shape):
        if input_shape is None:
            return None

        shape = input_shape
        if isinstance(shape, str):
            shape = ast.literal_eval(shape)

        if not isinstance(shape, (tuple, list)) or len(shape) != 3:
            raise ValueError(f"Invalid input-shape: {input_shape}. Expected (C,H,W)")

        _, h, w = shape
        h, w = int(h), int(w)
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid input-shape: {input_shape}. H and W must be > 0")
        return h, w

    @staticmethod
    def get_transform(dataset_id: str, is_train: bool, input_shape=None):
        if dataset_id not in DatasetConfig.TRANSFORMS:
            raise ValueError(f"Unknown dataset_id: {dataset_id}")

        if dataset_id == "uoft-cs/cifar10":
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
            steps = []

            hw = DatasetConfig._parse_hw_from_input_shape(input_shape)
            if hw is not None and hw != (32, 32):
                # Match the common MobileNet/CIFAR adaptation pipeline.
                steps.append(Resize(256, interpolation=InterpolationMode.BILINEAR))
                steps.append(CenterCrop(hw))

            if is_train:
                steps.append(RandomHorizontalFlip())

            steps.extend([
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])

            transform = Compose(steps)
        else:
            transform = DatasetConfig.TRANSFORMS[dataset_id][0 if is_train else 1]

        batch_key = DatasetConfig.BATCH_KEY[dataset_id]

        def apply_transforms(batch):
            batch[batch_key] = [transform(img.convert("RGB")) if isinstance(img, Image.Image) else transform(img) for
                                img in batch[batch_key]]
            return batch

        return apply_transforms
