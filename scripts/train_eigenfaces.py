#!/usr/bin/env python3
"""Train and evaluate an Eigenfaces (PCA) face recognizer with OpenCV.

Filenames must start with ``subjectNN`` (Yale Face Database convention); the
number NN is used as the class label. Files that don't match are skipped.

Usage:
    python scripts/train_eigenfaces.py \
        --train data/train --test data/test --output models/eigen_model.xml
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np

IMAGE_SIZE = (100, 100)  # must match the size used in face_recognition_node.py
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SUBJECT_RE = re.compile(r"subject(\d+)", re.IGNORECASE)


def load_dataset(directory: Path):
    """Return (images, labels) with labels parsed from the 'subjectNN' prefix."""
    images, labels = [], []
    for path in sorted(directory.iterdir()):
        match = SUBJECT_RE.match(path.name)
        if path.suffix.lower() not in IMAGE_EXTS or not match:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(cv2.resize(img, IMAGE_SIZE))
        labels.append(int(match.group(1)))
    return images, np.array(labels, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, default=Path("data/train"))
    parser.add_argument("--test", type=Path, default=Path("data/test"))
    parser.add_argument("--output", type=Path, default=Path("models/eigen_model.xml"))
    args = parser.parse_args()

    train_x, train_y = load_dataset(args.train)
    test_x, test_y = load_dataset(args.test)
    print(f"train: {len(train_x)} images, {len(set(train_y))} subjects")
    print(f"test:  {len(test_x)} images")

    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.train(train_x, train_y)

    # Only score test images whose subject exists in the training set.
    known = set(train_y.tolist())
    pairs = [(x, y) for x, y in zip(test_x, test_y) if y in known]
    skipped = len(test_x) - len(pairs)
    correct = sum(recognizer.predict(x)[0] == y for x, y in pairs)
    if pairs:
        print(f"accuracy: {correct / len(pairs):.3f} ({correct}/{len(pairs)})")
    if skipped:
        print(f"skipped {skipped} test image(s) from subjects not in the training set")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    recognizer.save(str(args.output))
    print(f"model saved to {args.output}")


if __name__ == "__main__":
    main()
