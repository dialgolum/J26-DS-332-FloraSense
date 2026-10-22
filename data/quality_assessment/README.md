# Buyer-Specific Export Quality Assessment Dataset

## Overview

This directory contains the image data and supporting annotations for Component 3: AI-Based Buyer-Specific Export Quality Assessment of the FloraSense research project.

The dataset is being prepared using ornamental plant images collected during field visits to Mike Flora (Pvt) Ltd. It will support the development and evaluation of computer-vision-based plant quality assessment methods.

## Dataset Structure

```text
quality_assessment/
├── original/
│   ├── Individual/
│   │   ├── P001/
│   │   └── P002/
│   ├── Batch/
│   │   └── P001/
│   └── Visual_Examples/
├── processed/
├── annotations/
│   └── quality_dataset.xlsx
└── README.md
```

The directories for additional plant types and visual examples will be added as required.

## Image Naming Convention

Images follow the naming format:

`PlantID_SampleID_View.jpg`

Examples:

- `P001_S001_front.jpg`
- `P001_S001_side.jpg`
- `P001_S001_root.jpg`
- `P001_S002_front.jpg`

Here, `P001` identifies the plant type, `S001` identifies an individual physical sample within that plant type, and the final part identifies the image view.

Batch images use identifiers such as `P001_B001.jpg`.

Multiple images belonging to the same physical sample retain the same Plant ID and Sample ID.

## Dataset Categories

**Individual Images:** Images of individual plant samples captured from different available views, including full-plant and root-area photographs.

**Batch Images:** Images containing multiple plant samples, intended for future investigation of batch-wise detection and quality assessment.

**Visual Examples:** Supplementary images illustrating specific quality conditions, where available.

## Quality Annotations

The planned annotation workbook, `quality_dataset.xlsx`, records plant information, sample measurements, applicable quality criteria, visual quality observations, image references, and expert assessment results.

The applicable criteria may vary depending on the plant type. For example, one plant type may require total leaf-count assessment, while another may require leaf-distribution assessment.

## Data Management

- Original images are preserved without modification.
- Processed images will be stored separately.
- Sample IDs are maintained consistently across related photographs.
- Unknown quality observations are distinguished from criteria that do not apply.
- Images and annotations will be linked using their identifiers.
- Data will be separated appropriately for training, validation, and testing to reduce data leakage.

## Current Status

Initial image collection and dataset organization are underway. The dataset currently focuses on one ornamental plant variety, with additional plant types and expert-verified annotations planned for subsequent stages.

The dataset is intended for research and development purposes. Storage and sharing of company-provided material must follow the permissions granted by Mike Flora (Pvt) Ltd.