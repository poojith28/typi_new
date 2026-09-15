from types import SimpleNamespace

from torchvision import transforms

from pycls.datasets.data import Data


def _cfg(dataset):
    return SimpleNamespace(
        DATA_LOADER=SimpleNamespace(NUM_WORKERS=0),
        DATASET=SimpleNamespace(
            NAME=dataset,
            ROOT_DIR=".",
            ACCEPTED=[dataset],
            AUG_METHOD="hflip",
        ),
        RANDAUG=SimpleNamespace(N=1, M=5),
    )


def _types(ops):
    return tuple(type(op) for op in ops)


def test_cifar_eval_has_no_random_spatial_transform():
    data = Data(_cfg("CIFAR10"))
    data.eval_mode = True
    kinds = _types(data.getPreprocessOps())
    assert transforms.RandomCrop not in kinds
    assert transforms.RandomHorizontalFlip not in kinds
    assert transforms.RandomResizedCrop not in kinds


def test_tinyimagenet_eval_uses_deterministic_resize():
    data = Data(_cfg("TINYIMAGENET"))
    data.eval_mode = True
    kinds = _types(data.getPreprocessOps())
    assert transforms.Resize in kinds
    assert transforms.RandomResizedCrop not in kinds
    assert transforms.RandomHorizontalFlip not in kinds


def test_training_transforms_remain_augmented():
    cifar = Data(_cfg("CIFAR100"))
    kinds = _types(cifar.getPreprocessOps())
    assert transforms.RandomCrop in kinds
    assert transforms.RandomHorizontalFlip in kinds

    tiny = Data(_cfg("TINYIMAGENET"))
    kinds = _types(tiny.getPreprocessOps())
    assert transforms.RandomResizedCrop in kinds
    assert transforms.RandomHorizontalFlip in kinds
