# Buyer-Specific Export Quality Assessment — Dataset Description

## Current C3 annotation clarification

Use [annotation_guidelines.md](annotation_guidelines.md) for current feature-label
interpretation. Leaf distribution means **qualifying branch origins on opposite
sides of the main stem**, not equal leaf counts or balanced foliage. Leaves
crossing the stem do not establish Both Sides. Occluded origins require expert
review without automatic annotation changes.

Released datasets and the researcher-confirmed Experiment 2 root corrections are
recorded in [dataset_version_history.md](dataset_version_history.md).

Current identifiers are P01–P08 with composite `(Plant_ID, Sample_ID)` identity.
The six current workbook sheets are Plant_Types, Requirement_Profiles, Samples,
Height_Assessments, Quality_Annotations, and Images. The planned schema and P001
examples below describe the historical design, not the current exact workbook.
Root and leaf labels must be versioned separately from overall quality; preserve
released experiment snapshots and grouped split assignments when labels change.

## 1. Dataset Overview

The dataset supports the development and experimental evaluation of Component 3: AI-Based Buyer-Specific Export Quality Assessment.

Initial data was collected from Mike Flora (Pvt) Ltd. and includes individual plant photographs, multiple-view images, and batch photographs of ornamental plants.

The dataset is being structured to support plant-specific quality assessment and configurable buyer/export-order requirements.

## 2. Data Collection

Plant photographs were captured under field conditions, including individual plant images placed against measurement references and batch images containing multiple plants.

Available images show different plant heights, orientations, leaf arrangements, and visible root/oasis characteristics.

The initial collection focuses on one plant variety. Additional varieties may be incorporated during later research stages.

## 3. Sample Identification

Each plant type receives a unique identifier:

- `P001` — Plant Type 1
- `P002` — Plant Type 2

Each physical sample within a plant type receives a separate identifier:

- `S001` — Sample 1
- `S002` — Sample 2

The combination of Plant ID and Sample ID uniquely identifies a physical sample.

Example: `P001_S001`

Multiple images of the same sample retain the same identifier.

## 4. Dataset Annotation Structure

The planned annotation workbook contains the following sheets.

### 4.1 Plant_Types

Records plant identifiers, variety names, and the quality characteristics applicable to each plant type.

### 4.2 Requirement_Profiles

Records configurable buyer/export-order requirements, including plant height ranges, mode requirements, and applicable quality thresholds.

### 4.3 Samples

Records individual sample identifiers, plant type, reference measurements, observed mode, and available expert assessment results.

### 4.4 Quality_Annotations

Records observed quality characteristics for each sample, including:

- Stem count
- Total leaf count
- Root distribution
- Leaf distribution, where applicable
- Overall visual quality
- Oasis damage condition

For the initial plant type, leaf count is assessed at the whole-sample level. The provisional minimum total leaf requirement is calculated as:

`Minimum Total Leaves = Stem Count × 4`

This simplified assessment rule will be validated against the company's actual inspection practice.

### 4.5 Images

Links each image to its plant type and physical sample using filenames and relative file paths.

A sample may have multiple images, and root close-up images are optional.

### 4.6 Visual_Examples

Stores references and labels for supplementary visual-quality examples that cannot be associated with a specific physical sample.

## 5. Quality Assessment Requirements

The quality assessment process distinguishes between two categories.

**Buyer/Export-Order Requirements**

These may include plant height ranges, plant size, and order-specific modes such as one or two oasis units.

**Plant-Specific Quality Criteria**

These may include minimum leaf count, leaf distribution, root distribution, oasis condition, and overall plant appearance.

The applicable criteria differ across plant types.

## 6. Annotation Values

Quality annotations will use consistent values such as:

- `Acceptable`
- `Unacceptable`
- `Yes`
- `No`
- `Unknown`
- `N/A`

`Unknown` indicates that a required observation has not been established.

`N/A` indicates that a particular quality criterion does not apply to the plant type.

Where possible, visual quality labels will be verified using expert assessments from Mike Flora.

## 7. Dataset Preparation

Planned preprocessing activities include:

- Image format verification
- Image resizing where required
- Image orientation correction
- Image quality checks
- Linking images with annotation records
- Preparation of suitable experimental datasets

Original images will remain unchanged.

## 8. Dataset Splitting and Evaluation

For model development, training, validation, and testing partitions will be prepared based on the available labelled data.

Multiple images belonging to the same physical sample must remain within the same partition to prevent data leakage.

Batch images and crops derived from the same source image must also be handled together.

Evaluation procedures and dataset split proportions will be finalized according to the number of available independent samples.

## 9. Current Limitations

The initial dataset has several limitations:

- Limited representation of plant varieties.
- Limited availability of expert-labelled unacceptable-quality examples.
- Different numbers of images per physical sample.
- Some visual characteristics may not be fully observable from a single image.
- Batch images may contain overlapping leaves and roots.
- Reference measurements and quality labels may be incomplete for some samples.

These limitations will be considered when selecting assessment techniques and interpreting experimental results.

## 10. Planned Use

The dataset will support experiments in plant measurement, leaf counting, root/oasis quality assessment, and evaluation of configurable export quality decisions.

Batch-image processing may be investigated after establishing the initial individual-plant assessment approach.
