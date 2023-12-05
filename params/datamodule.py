import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from lib.data.cfg.local import LocalNaturalDatasetCfg
from lib.data.datamodules.vanilla import AUDataModule

project = "disfa"

aus = [
    "AU1",
    "AU2",
    "AU4",
    "AU5",
    "AU6",
    "AU9",
    "AU12",
    "AU15",
    "AU17",
    "AU20",
    "AU26",
]

TRAIN_LABELED = [
    LocalNaturalDatasetCfg(
        name="disfa",
        root="/data",
        aus=aus,
        crops_dir="/data/cropped_images",
        labels_filename="df_proc_tmp_train.csv",
    )
]

TRAIN_UNLABELED = []

VAL_DATASETS = [
    LocalNaturalDatasetCfg(
        name="disfa",
        root="/data",
        aus=aus,
        crops_dir="/data/cropped_images",
        labels_filename="df_proc_tmp_test.csv",
    )
]

# To enable WandB
# run = wandb.init(project=project)
# for ds in TRAIN_LABELED + TRAIN_UNLABELED + VAL_DATASETS:
#     ds.register_art(run, fetch=False)

image_size = 224
logits_per_class = 2
# num_aus = len(TRAIN_SYNTH[0].aus)
num_aus = len(TRAIN_LABELED[0].aus)

mean = IMAGENET_DEFAULT_MEAN
std = IMAGENET_DEFAULT_STD

weak_transforms = A.Compose(
    [
        A.SmallestMaxSize(image_size + 20),
        A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
)

medium_transforms = A.Compose(
    [
        A.SmallestMaxSize(image_size + 20),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=30),
        A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(),
        A.SomeOf(
            [
                A.OneOf(
                    [
                        A.ImageCompression(),
                        A.ISONoise(),
                        A.GaussNoise(),
                    ]
                ),
                A.OneOf(
                    [
                        A.MotionBlur(),
                        A.MedianBlur(blur_limit=3),
                        A.Blur(blur_limit=3),
                    ]
                ),
                A.SomeOf(
                    [
                        A.CLAHE(),
                        A.Equalize(),
                        A.Sharpen(),
                        A.RandomBrightnessContrast(),
                        A.RandomGamma(),
                        A.HueSaturationValue(),
                        A.RGBShift(),
                    ],
                    n=3,
                ),
            ],
            n=2,
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
)

strong_transforms = A.Compose(
    [
        A.SmallestMaxSize(image_size + 20),
        A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.2, rotate_limit=30),
        A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(),
        A.SomeOf(
            [
                A.OneOf(
                    [
                        A.ImageCompression(),
                        A.ISONoise(),
                        A.GaussNoise(),
                    ]
                ),
                A.OneOf(
                    [
                        A.MotionBlur(),
                        A.MedianBlur(blur_limit=3),
                        A.Blur(blur_limit=3),
                    ]
                ),
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=2),
                        A.Equalize(),
                        A.Sharpen(alpha=(0.25, 0.75), lightness=(0.25, 0.75)),
                        A.RandomBrightnessContrast(
                            brightness_limit=(0.25, 0.75), contrast_limit=(0.25, 0.75)
                        ),
                        A.Emboss(),
                        A.HueSaturationValue(),
                    ]
                ),
                A.OneOf(
                    [
                        A.Solarize(),
                        A.Posterize(),
                        A.ColorJitter(),
                        A.RGBShift(
                            r_shift_limit=15, g_shift_limit=15, b_shift_limit=15
                        ),
                    ]
                ),
                A.OneOf(
                    [
                        A.Superpixels(),
                        A.RandomRain(),
                    ]
                ),
            ],
            n=3,
        ),
        A.CoarseDropout(max_holes=2, max_height=50, max_width=50, fill_value=125),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
)

val_transforms = A.Compose(
    [
        A.SmallestMaxSize(image_size),
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
)

datamodule = AUDataModule(
    dataset_cfg={
        "train": TRAIN_LABELED,
        "val": VAL_DATASETS,
    },
    image_size=image_size,
    logits_per_class=logits_per_class,
    train_transforms=medium_transforms,
    val_transforms=val_transforms,
    batch_size=150,
    num_workers=4,
)
