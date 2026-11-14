# Eagen — Real-time Face Recognition with ROS 2 and Eigenfaces

A ROS 2 pipeline that captures webcam frames, detects faces with a Haar cascade, and identifies them with a classic **Eigenfaces (PCA)** recognizer trained with OpenCV.

```
webcam_publisher ──/webcam_image──▶ face_recognition_node ──/faces──▶ (any subscriber)
   (sensor_msgs/Image)               Haar detect → resize 100×100        face_msgs/Faces
                                     → Eigenfaces predict → overlay
```

## Features

- **Training/evaluation script** for Eigenfaces with per-subject labels parsed from filenames.
- **Two ROS 2 nodes**: `webcam_publisher` (camera → `/webcam_image`) and `face_recognition_node` (detection + recognition → `/faces`, plus a live overlay window).
- **Custom messages** (`face_msgs/Faces`, `face_msgs/Facebox`) carrying bounding box, predicted label and confidence (distance) per face.
- Configurable model path via ROS parameter.

## Repository layout

```
├── src/
│   ├── face_msgs/          # custom ROS 2 message definitions
│   └── face_recognition/   # webcam publisher + recognition node
├── scripts/
│   └── train_eigenfaces.py # train + evaluate, writes models/eigen_model.xml
├── models/                 # trained model (OpenCV XML)
├── data/
│   ├── train/, test/       # Yale-style face images (subjectNN_*.jpg)
│   └── att_faces/          # AT&T face database
└── requirements.txt
```

## Getting started

**Requirements:** Ubuntu with ROS 2 (Humble or later), Python 3.10+, a webcam.

```bash
# 1. Python deps for training
pip install -r requirements.txt

# 2. Train and evaluate (writes models/eigen_model.xml)
python scripts/train_eigenfaces.py --train data/train --test data/test

# 3. Build the ROS 2 workspace
source /opt/ros/<distro>/setup.bash
colcon build
source install/setup.bash

# 4. Run (two terminals, from the repo root)
ros2 run face_recognition webcam_publisher.py
ros2 run face_recognition face_recognition_node.py --ros-args -p model_path:=models/eigen_model.xml
```

Inspect output with `ros2 topic echo /faces`.

## How it works

1. **Detection** — `cv2.CascadeClassifier` (Haar frontal face) on the grayscale frame.
2. **Recognition** — each face crop is resized to 100×100 and projected onto the Eigenfaces subspace; the nearest training sample gives the label, and the distance is reported as `confidence` (**lower = more similar**).
3. **Decision** — a face is shown as "Recognized" when distance < 4100 (an empirical threshold; tune it for your data).

## Datasets

- [Yale Face Database](http://vision.ucsd.edu/datasets/yale_face_dataset_original/yalefaces.zip) subset (`subjectNN_<condition>`: lighting, glasses, expressions).
- [AT&T (ORL) Database of Faces](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html).

Both are used for research/educational purposes under their own terms.

## Limitations & next steps

- Eigenfaces is sensitive to lighting, pose and alignment; a natural next step is comparing against a deep embedding model (e.g. FaceNet/ArcFace) on the same split.
- Add a results table (accuracy per condition) and a demo GIF.
- Unit tests for the data loader and a launch file starting both nodes.

## License

MIT — see [LICENSE](LICENSE).
