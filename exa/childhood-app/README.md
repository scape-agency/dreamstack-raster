# Childhood App

Object detection and extraction pipeline for processing image folders.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install ultralytics
   ```

2. **Add images:**
   Place your images in the `images/` folder.

3. **Run:**
   ```bash
   python main.py
   ```

4. **Check output:**
   Results will be in the `output/` folder.

## Output Structure

```
output/
├── image_0001/
│   ├── metadata.json    # Description and object list
│   ├── dog_1.png        # Extracted with alpha channel
│   ├── person_1.png
│   └── bicycle_1.png
├── image_0002/
│   ├── metadata.json
│   └── car_1.png
└── ...
```

## Example metadata.json

```json
{
  "source_image": "photo_001.jpg",
  "processed_at": "2026-02-20T14:32:00.000000",
  "description": "Image contains 3 objects: dog, bicycle, person",
  "image_size": {
    "height": 1080,
    "width": 1920
  },
  "num_objects": 3,
  "labels": ["bicycle", "dog", "person"],
  "objects": [
    {
      "label": "dog",
      "confidence": 0.92,
      "bbox": [100, 150, 200, 250],
      "file": "dog_1.png"
    },
    {
      "label": "person",
      "confidence": 0.87,
      "bbox": [400, 100, 150, 400],
      "file": "person_1.png"
    },
    {
      "label": "bicycle",
      "confidence": 0.78,
      "bbox": [600, 300, 300, 200],
      "file": "bicycle_1.png"
    }
  ]
}
```

## CLI Options

```
usage: main.py [-h] [--input INPUT] [--output OUTPUT] [--model MODEL]
               [--confidence CONFIDENCE] [--margin MARGIN] [--recursive]
               [--copy-source] [--device {auto,cpu,cuda,mps}] [--verbose]

options:
  -h, --help            show this help message and exit
  --input, -i           Input directory (default: ./images)
  --output, -o          Output directory (default: ./output)
  --model, -m           YOLO model name (default: yolov8n-seg)
  --confidence, -c      Min detection confidence (default: 0.5)
  --margin              Margin around objects in px (default: 10)
  --recursive, -r       Process subdirectories
  --copy-source         Copy source images to output
  --device              Inference device: auto, cpu, cuda, mps
  --verbose, -v         Verbose output
```

## Models

Available YOLO segmentation models (smallest to largest):

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolov8n-seg` | ~3MB | Fastest | Basic |
| `yolov8s-seg` | ~11MB | Fast | Good |
| `yolov8m-seg` | ~26MB | Medium | Better |
| `yolov8l-seg` | ~46MB | Slow | High |
| `yolov8x-seg` | ~69MB | Slowest | Best |

For Mac M2, `yolov8n-seg` or `yolov8s-seg` are recommended.

## Device Selection

- **auto** (default): Uses MPS on Mac, CUDA on NVIDIA, CPU otherwise
- **mps**: Force Apple Metal (Mac M1/M2/M3)
- **cuda**: Force NVIDIA GPU
- **cpu**: Force CPU (slowest)
