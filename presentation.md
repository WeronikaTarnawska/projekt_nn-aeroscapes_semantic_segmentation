---
title: Semantic segmentation - Aeroscapes dataset
authors: 
  - Kacper Chmielewski
  - Patryk Maciąg
  - Adrian Walczak
  - Weronika Tarnawska
options:
  implicit_slide_ends: true
  end_slide_shorthand: true
  command_prefix: "cmd: "
  image_attributes_prefix: ""
theme:
  override:
  max_rows_alignment: center
---

Cel
===

Dane wejściowe:
Zbiór danych 𝒟 = {(I₁, Y₁), (I₂, Y₂), …, (Iₙ, Yₙ)}, gdzie:
○ Iᵢ ∈ ℝ^(H×W×3) są obrazami RGB z drona
○ Yᵢ ∈ ℂ^(H×W) są mapami etykiet pikselowych
○ ℂ = {l₁, l₂, …, l_k} to zbiór klas semantycznych
Cel:
Funkcję f: ℝ^(H×W×3) → ℂ^(H×W)
Warunek:
f przypisuje każdemu pikselowi obrazu z drona odpowiednią klasę semantyczną
f maksymalizuje prawdopodobieństwo poprawnej klasyfikacji: P(f(I)[u,v] = Y[u,v])
Model radzi sobie z:
zmienną skalą obiektów (różne wysokości lotu)
małymi obiektami (np. ludzie, rowery)
niezbalansowanymi klasami (rzadkie obiekty)


Dataset
=======

[aeroscapes](https://www.kaggle.com/datasets/kooaslansefat/uav-segmentation-aeroscapes)

- drone images
- expected color maps (11 classes)
- goal: assign correct class to each pixel of the image


EDA results
============================

- Images: all 1280×720 RGB — single resize policy works; aspect ratio 16:9.
- Masks: mode `L` (single-channel index map), values exactly `{0, ..., 11}`. **No color decoding needed**, no `255` ignore label to mask out.
- Mask loading: `np.array(Image.open(p))` returns class indices directly — cast to `torch.long` for `CrossEntropyLoss`.
- Resize interpolation: **`BILINEAR` for images, `NEAREST` for masks** (any blending of class indices would create invalid intermediate values like `4.7`).
- **Normalization decision.** Aeroscapes means `[0.44, 0.51, 0.46]` vs ImageNet `[0.49, 0.46, 0.41]` - For preprocessing: **use ImageNet mean/std** when training with a pretrained backbone - the differences are small and the backbone expects that input distribution.

EDA - Class imbalance
===

- **(a) Pixel share** — fraction of all training pixels belonging to each class. Drives loss weighting.
- **(b) Image presence** — fraction of training images that contain the class at all. Drives class-balanced sampling.
- **(c) Mean coverage when present** — when the class IS in an image, what fraction does it cover. Distinguishes *rare* (few images) from *small* (many images, tiny objects) — these need different mitigations.

![width:100%](./tmp/6_eda_clacc_imbalance.png)

EDA - class imbalance
===

- **3 dominant classes**: vegetation (35.4% pixels / 92% images), road (31.6% / 87.5%), background (23.0% / 100% — present in every single image). Together ~90% of all pixels. Without weighting, the loss will be driven almost entirely by these.
- **Small but ubiquitous** (low pixel share, high presence): person (0.43% / 85%), bike (0.06% / 42%), obstacle (0.6% / 66%). These appear in many scenes but as tiny objects — IoU will collapse without pixel-level loss weighting.
- **Genuinely rare** (low on both): boat (0.02% / 1.9%), animal (0.07% / 2.8%), drone (0.03% / 10%). Loss weighting alone is not enough — also need `WeightedRandomSampler` so the model sees them more than a handful of times per epoch.
- **Bursty**: sky (only 15% of images, but covers ~31% when present — open-aerial shots). Augmentation crops should not throw away the few sky-containing images.

**Preprocessing implication**: combine median-frequency loss weights *and* a sampler that upweights images containing rare classes. Background is huge but not informative — consider downweighting it explicitly in the loss (or using `ignore_index`).


EDA - class coocurence
===

**Co-occurrence findings:**
- `background` is in **100%** of all images regardless of conditioning class — it acts as a true "everything else" label. Downweighting it (or using `ignore_index=0`) will not lose information about other classes.
- **Boat** is highly isolated: when boat is present, road appears in only 2% of images and vegetation in 16% (vs. 87–92% baseline). The model could shortcut "water-like backdrop → boat" — worth checking on validation.
- **Animal** rarely co-occurs with road (5.5%) but almost always with construction (96%) and vegetation (96%) — rural scenes only.
- **Drone** has the most distinctive context: 35% with sky vs. 15% baseline (drones photographed against sky), and lower co-occurrence with bike/animal.
- **Sky** rarely contains bike (6%) — bike scenes are typically low-altitude, sky-less.

These are **shortcut risks**, not preprocessing changes — but they motivate keeping color jitter and contrast augmentations to discourage backdrop-based shortcuts.

EDA - class weights
===

![width:100%](./tmp/table3_class_weights.png)

Summary of findings → preprocessing decisions
===

Each row links one EDA finding to a concrete choice for `AeroScapesDataset` / `AeroScapesDataModule`:

| Finding | Decision |
|---|---|
| 3269 imgs ↔ 3269 masks, all paired by stem | Trust the structure — no filtering needed |
| All images 1280×720 RGB uint8 | Single resize policy; aspect ratio 16:9 (preserve or square-crop) |
| Masks are mode `L`, values exactly `{0..11}`, no `255` | Load as `np.array(...).astype(np.int64)` — no color decoding, no ignore mask |
| Mask values are class indices, not continuous | Resize: `BILINEAR` for image, **`NEAREST` for mask** |
| Train RGB mean `[0.44, 0.51, 0.46]` close to ImageNet `[0.49, 0.46, 0.41]` | Use ImageNet mean/std (we'll use a pretrained backbone) |
| 3 classes (vegetation, road, background) hold ~90% of pixels; bottom 4 (boat, animal, drone, bike) hold <0.2% combined | Median-frequency class weights for `CrossEntropyLoss`; consider downweighting / `ignore_index` for background |
| boat (1.9%), animal (2.8%), drone (10%) appear in very few *images* | `WeightedRandomSampler` on top of loss weighting — pixel weights alone don't help if a class is missing from most batches |
| person, bike, obstacle: high presence (42–85%) but tiny pixel share | Per-pixel loss weighting will already help; avoid heavy downsampling that erases small objects (keep ≥ 360p, prefer crops over resize for fine detail) |
| Median 6 classes per image (range 2–9) | Crops up to ~1/8 of width still see enough class diversity — random-crop augmentation is safe |
| Use the official `ImageSets/trn.txt` / `val.txt` split | No `random_split` — keeps results comparable to published numbers |

These map directly onto the pieces the next stage (`src/datasets/aeroscapes.py` + DataModule) needs to implement.

Baseline
===

TODO description

```
import fiddle as fdl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config.aeroscapes_constants import (
    CLASS_NAMES,
    IGNORE_INDEX_BACKGROUND,
    MEDIAN_FREQ_WEIGHTS,
    NUM_CLASSES,
)
from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.config.schemas import ExperimentConfig, TrainingConfig
from src.datasets.aeroscapes import AeroScapesDataModule
from src.models.architectures.torchvision_segmentation import TorchVisionSegmentation
from src.models.segmentation_model import SegmentationModel
from src.utils.callbacks import LogPredictionsCallback


def build_config() -> fdl.Config[ExperimentConfig]:
    max_epochs = 30

    # weights=None: ImageNet-pretrained backbone + fresh head sized to NUM_CLASSES.
    # Switch to weights="DEFAULT" to start from full COCO-pretrained model
    # (final conv gets surgically swapped). Flip freeze_backbone for the
    # frozen-feature baseline used in the assignment.
    architecture = fdl.Config(
        TorchVisionSegmentation,
        name="deeplabv3_resnet50",
        num_classes=NUM_CLASSES,
        weights=None,
        freeze_backbone=False,
    )

    # crop_size=512: square random crops keep small classes (person/bike/obstacle)
    # at native pixel scale while still allowing batch>1. Validation runs at
    # native 1280x720 (eval_size=None) for paper-comparable mIoU.
    data_module = fdl.Config(
        AeroScapesDataModule,
        "data/aeroscapes",
        batch_size=8,
        val_batch_size=2,
        crop_size=512,
    )

    # class_weights: median-frequency from EDA (notebooks/eda.ipynb) — keeps the
    # loss from being dominated by vegetation/road/background. Plain list of
    # floats: SegmentationModel converts to tensor internally (Fiddle-friendly,
    # avoids wrapping torch.tensor which has no inspectable signature).
    # ignore_index=0: background is ~23% of pixels but carries no semantic info,
    # so exclude it from loss AND metrics rather than just downweighting it.
    model = fdl.Config(
        SegmentationModel,
        model=architecture,
        num_classes=NUM_CLASSES,
        class_weights=list(MEDIAN_FREQ_WEIGHTS),
        ignore_index=IGNORE_INDEX_BACKGROUND,
        lr=1e-4,
        weight_decay=1e-4,
    )

    wandb_logger = fdl.Partial(
        WandbLogger,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
    )

    # save_last=True is required by train_model.py's resume mechanism (it looks
    # for last.ckpt first). monitor="val/miou" picks the best checkpoint by mIoU.
    checkpoints_callback = fdl.Partial(
        ModelCheckpoint,
        monitor="val/miou",
        mode="max",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
    )

    return fdl.Config(
        ExperimentConfig,
        "deeplabv3_resnet50_aeroscapes",
        model,
        data_module,
        training_cfg=fdl.Config(
            TrainingConfig,
            wandb_logger,
            checkpoints_callback,
            max_epochs,
            callbacks=[
                fdl.Config(
                    LogPredictionsCallback,
                    num_samples=8,
                    class_names=CLASS_NAMES,
                ),
            ],
        ),
    )
```

Reference solutions
===

**Semantic segmentation** (comparable to our task):

![width:100%](./tmp/table1.png)

**Classification / detection** (mIoU not applicable): EfficientNet ~98%, YOLOv8 74%, etc.

The highest segmentation scores (~0.93) come from a strong encoder (DPN107) and plain
CE, without explicit imbalance handling — but also at **256×256** and with **11 output
classes** (background omitted from the model). We validate at **native 1280×720**
(512 px crops in training) with `ignore_index=0`, so the headline numbers above are
only loosely comparable.

Our two methods
===

![width:100%](./tmp/table2.png)

**Method A** targets class imbalance. Architecture matches the CE baseline; only the
loss changes to `DiceCELoss = weighted CE + Dice`. CE is pixel-weighted, so rare classes
barely affect the gradient. Dice averages per-class overlap, giving small objects
comparable influence to large ones.

**Method B** targets overall mIoU: ResNet101, full COCO pretraining, the same Dice+CE
loss as A, and a cosine LR schedule. Capacity and pretraining differ; the loss does not.


Results
=======

Best-epoch validation mIoU (checkpoint selection metric), 11 foreground classes,
full-resolution 1280×720 validation.

```
run                           best mIoU  best epoch
----------------------------------------------------
Baseline CE (r50)                 0.672          22
A: Dice+CE (r50)                  0.709           8
B: r101 COCO+Dice+CE              0.694          21

A: Dice+CE (r50): +0.037 vs baseline
B: r101 COCO+Dice+CE: +0.022 vs baseline
```

Swapping the loss alone (Method A) gave the largest gain (**+0.037 mIoU**) and
briefly outperformed the heavier ResNet101 setup. On this dataset, the Dice term
mattered more than adding model capacity.

Results
=======

![width:100%](./tmp/1_val_miou_per_epoch.png)

Results
=======

![width:100%](./tmp/2_val_loss_noise.png)

Results
=======

![width:100%](./tmp/3_per_class_iou.png)

Results
=======

![width:100%](./tmp/4_A_hard_easy_classes.png)

Results
=======

- Hard classes improve but remain well below road/vegetation (~0.95). At Method A's
  best epoch: obstacle 0.235→0.266, bike 0.316→0.437, animal 0.355→0.397. Method B
  does better on obstacle/animal (0.284 / 0.451) at some cost to bike and person.
- Easy classes stay stable — the right-hand panel is nearly flat. Early val-loss
  spikes are batch noise (small val set + Dice), not a real regression; trust
  per-class IoU over loss alone.
- Background is excluded from metrics (`ignore_index=0`); see section 4 for how
  that shows up visually.
- The two models excel on different classes (A: person/bike/car; B: obstacle/animal),
  which motivates the fusion idea in section 5.

Qualitative analysis - why the hard classes are hard
===

We run both checkpoints on validation images that contain obstacle, bike, or animal
and compare predictions side by side. Requires checkpoints in `logs/` and the dataset
under `data/aeroscapes/`.

**Background (black in GT).** Class 0 is excluded from both loss and mIoU
(`ignore_index=0`; references use the same convention with 11 scored classes). The
network still has 12 outputs, but receives no gradient on GT-background pixels, so
predictions often contain no black at all — those regions get assigned to foreground
classes instead. That can look wrong on the figure, but those pixels are not included
in val mIoU.

Qualitative analysis - why the hard classes are hard
===

![width:100%](./tmp/5_quantitative_analysis.png)

What stands out in the images
===

- **Small, thin objects.** Bikes and obstacles span only a few pixels; a single-pixel
  boundary shift costs a large fraction of IoU even when the mask looks reasonable.
- **Vague class boundary.** *Obstacle* covers poles, rocks, debris, fences — little
  shared appearance. Low IoU here reflects ambiguous labels as much as frequency.
- **Scale range.** Altitude varies widely, so the same object type can occupy very
  different pixel counts; small instances are easy to lose after downsampling.

The two models often err on different parts of the same scene (see per-class IoU
under each column), matching the complementarity in the bar chart above.

Next steps for the hard classes
===

1. **Larger crops / higher resolution** — most direct way to preserve thin objects;
   limited by GPU memory.
2. **Loss tuning** — increase `dice_weight`, or add Focal loss for hard pixels;
   [`losses.py`](../src/models/losses.py) is set up to extend.
3. **Copy-paste augmentation** for bike / obstacle / animal — more training exposure
   at usable pixel scale without waiting for rare scenes.
4. **Two-model fusion (longer term).** Train a hard-class specialist (strong Dice +
   class weights, accepting weaker easy classes) and combine it with the generalist
   run — e.g. per-class confidence or a lightweight router. Methods A and B already
   win on different classes, so there is room to merge their strengths.