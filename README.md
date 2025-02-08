# 3D Map Stitching using Automatic Overlapping Area Identification (NOT COMPLETED YET)

## Overview
This repository contains the implementation of the **"automatic overlaping area identification"** for 3D map stitching algorithm. The method improves computational efficiency by identifying overlapping areas automatically instead of processing the entire map.

## Status
🚧 **Work in Progress** 🚧
This repository is currently under development. The full code will be added soon. Stay tuned!

## Features
- **Automatic Overlapping Area Identification**: Uses **DBSCAN clustering** and **template matching** to detect overlapping areas between partial maps.
- **Binary Search Optimization**: Fine-tunes clustering parameters for improved performance.
- **Tested on Large-Scale 3D Maps**: Evaluated using the KITTI dataset and real-world 3D maps.

## Installation
Clone the repository:
```sh
git clone https://github.com/ahsanfs/auto-overlapping_area_identification.git
```

### Dependencies
Ensure you have Python and required libraries installed:
```sh
pip3 install numpy open3d scipy matplotlib
```

## Usage
Run the main script to perform 3D map stitching:
```sh
python3 main.py --input path/to/pointcloud1.pcd path/to/pointcloud2.pcd --output stitched_map.ply
```

## Algorithm Workflow
1. **Preprocessing**: Noise filtering using intensity thresholding.
2. **Clustering**: Apply DBSCAN to segment point clouds.
3. **Template Matching**: Compute correlation scores to identify corresponding clusters.
4. **Binary Search Optimization**: Fine-tune clustering parameters for optimal results.
5. **Map Stitching**: Use map-merge-3D algorithm to stitch maps together.


## Dataset
- **KITTI Dataset**: Used for benchmarking large-scale map stitching.
- **Real-world Maps**: NTUT and NCL maps were tested to validate the approach.


## Contact
For any inquiries, please reach out to `muhammadahsanfs@gmail.com`.

