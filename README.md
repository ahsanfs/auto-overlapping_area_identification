# 3D Map Stitching using Automatic Overlapping Area Identification

## Overview
This repository contains the implementation of the 3D map stitching algorithm proposed in the paper **"An Efficient Large-Scale 3D Map Stitching Algorithm using Automatic Overlapping Area Identification"**. The method improves computational efficiency by identifying overlapping areas automatically instead of processing the entire map.

## Features
- **Automatic Overlapping Area Identification**: Uses **DBSCAN clustering** and **template matching** to detect overlapping areas between partial maps.
- **Binary Search Optimization**: Fine-tunes clustering parameters for improved performance.
- **Efficient Map Stitching**: Reduces computation time by 38.64% compared to traditional methods.
- **Tested on Large-Scale 3D Maps**: Evaluated using the KITTI dataset and real-world 3D maps.

## Installation
Clone the repository:
```sh
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

### Dependencies
Ensure you have Python and required libraries installed:
```sh
pip install numpy open3d scipy matplotlib
```

## Usage
Run the main script to perform 3D map stitching:
```sh
python main.py --input path/to/pointcloud1.ply path/to/pointcloud2.ply --output stitched_map.ply
```

## Algorithm Workflow
1. **Preprocessing**: Noise filtering using intensity thresholding.
2. **Clustering**: Apply DBSCAN to segment point clouds.
3. **Template Matching**: Compute correlation scores to identify corresponding clusters.
4. **Binary Search Optimization**: Fine-tune clustering parameters for optimal results.
5. **Map Stitching**: Use map-merge-3D algorithm to stitch maps together.

## Results
| Method               | Translation Error (m) | Rotation Error (°) | Time Reduction (%) |
|----------------------|---------------------|-------------------|------------------|
| Entire Maps         | 0.1819              | 0.2103            | 0%               |
| Human Selection     | 0.4278              | 0.7123            | 22.26%           |
| Proposed Algorithm  | 0.1723              | 0.1763            | 38.64%           |

## Dataset
- **KITTI Dataset**: Used for benchmarking large-scale map stitching.
- **Real-world Maps**: NTUT and NCL maps were tested to validate the approach.

## Citation
If you find this work useful, please cite:
```bibtex
@article{YourPaper2024,
  author    = {Hsien-I Lin and Muhammad Ahsan Fatwaddin Shodiq and An-Kai Jeng and Chun-Wei Chang},
  title     = {An Efficient Large-Scale 3D Map Stitching Algorithm using Automatic Overlapping Area Identification},
  journal   = {IEEE Access},
  year      = {2024},
  doi       = {10.1109/ACCESS.2024.0322000}
}
```

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contact
For any inquiries, please reach out to `your.email@example.com`.

