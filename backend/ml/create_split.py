from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"

SOURCE = DATASET / "train"
VALIDATION = DATASET / "validation"

SEED = 42
VALIDATION_RATIO = 0.10

random.seed(SEED)

classes = ["NORMAL", "PNEUMONIA"]

for class_name in classes:

    source_dir = SOURCE / class_name
    validation_dir = VALIDATION / class_name

    validation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    images = [
        p for p in source_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png"
        ]
    ]

    random.shuffle(images)

    validation_count = int(
        len(images) * VALIDATION_RATIO
    )

    validation_images = images[
        :validation_count
    ]

    print(
        f"{class_name}: "
        f"{len(images)} total, "
        f"{validation_count} validation"
    )

    for image in validation_images:

        destination = validation_dir / image.name

        shutil.move(
            str(image),
            str(destination)
        )

print("\nValidation split created successfully.")