# Component 3: Buyer-Specific Export Quality Assessment — Architecture

## 1. Overview

The Buyer-Specific Export Quality Assessment component is responsible for assessing ornamental plant quality against applicable plant-specific quality criteria and configurable buyer/export-order requirements.

The component aims to combine computer-vision-based quality assessment with a configurable decision engine to support consistent export quality decisions.

## 2. Proposed System Workflow

The proposed assessment workflow consists of the following stages:

1. **Input Acquisition:** Receive plant images, plant type information, and buyer/export-order requirements.
2. **Image Preprocessing:** Prepare images for computer-vision analysis.
3. **Quality Characteristic Assessment:** Extract or assess relevant characteristics, including plant height, leaf count, root distribution, oasis condition, and other applicable visual quality attributes.
4. **Requirement Configuration:** Retrieve plant-specific quality criteria and buyer/export-order requirements.
5. **Requirement Matching:** Compare assessed characteristics against the applicable quality criteria.
6. **Quality Decision:** Generate PASS, FAIL, or MANUAL REVIEW decisions with supporting reasons.

## 3. Main Modules

### 3.1 Quality Requirement Configuration

Maintains configurable quality requirements associated with plant types and buyer/export orders.

Buyer/order requirements may include plant height ranges and packing modes, while plant-specific criteria may include leaf count, root distribution, leaf distribution, oasis condition, and overall visual quality.

Not every quality criterion applies to every plant type.

### 3.2 Computer Vision Assessment

This module is planned to investigate image-based assessment techniques for relevant physical and visual quality characteristics.

Potential tasks include:

- Plant height estimation
- Leaf counting
- Root distribution assessment
- Oasis damage assessment
- Plant-specific visual quality assessment

The techniques will be selected and evaluated based on dataset availability and experimental results.

### 3.3 Quality Decision Engine

The decision engine compares observed quality characteristics against the selected requirement profile.

It is designed to generate:

- **PASS:** All applicable requirements are satisfied.
- **FAIL:** At least one applicable requirement is not satisfied.
- **MANUAL REVIEW:** Available information is insufficient for a reliable acceptance decision.

The decision output will include the reasons associated with failed or incomplete assessments.

### 3.4 Data Management

Plant images, sample information, quality annotations, and requirement profiles are maintained in structured data sources.

The initial prototype uses an Excel-based annotation dataset and JSON-based quality requirement configurations.

## 4. Proposed Integration

The component will be developed as an individual service within the FloraSense repository.

Planned integration includes:

- Receiving plant and order information from the shared system.
- Accessing relevant plant records and quality requirements.
- Processing plant quality information.
- Returning assessment decisions and supporting explanations.
- Making results available to the shared frontend through the agreed service/API integration approach.

Integration details will be finalized with the research team.

## 5. Initial Implementation Scope

The initial implementation focuses on:

- Dataset organization and annotation preparation.
- Plant-specific and buyer/order requirement configuration.
- Development of a Python-based quality decision engine.
- Initial image-processing experiments using the available plant images.

The initial prototype will use manually recorded measurements and expert quality observations where automated assessment is not yet available.

## 6. Future Development

Planned development includes computer-vision-based quality characteristic extraction, evaluation of visual quality assessment methods using limited labelled data, and integration with the complete FloraSense system.

Batch-wise plant assessment may also be investigated as an extension of individual-plant quality assessment.