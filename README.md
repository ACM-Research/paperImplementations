![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Summer Paper Implementation for Cancer Imaging Optimized via GANs

# M3Dsynth: A Large Dataset for Medical Image Manipulation Detection

## Introduction

This paper introduces M3Dsynth, a large dataset of manipulated 3D medical images, focusing on computed tomography (CT) lung scans with artificially injected or removed lung cancer nodules.

### Key Features
- Over 8,000 manipulated CT scans
- Three different generative methods:
  1. Two based on Generative Adversarial Networks (GANs)
  2. One using Diffusion Models (DMs)

### Motivation
To address the lack of large, curated datasets for:
- Developing methods to detect manipulations in medical images
- Benchmarking existing detection techniques

This is particularly important given recent advances in image synthesis techniques.

## Findings and Contributions

### 1. Vulnerability of Diagnostic Tools
- Manipulated images successfully fooled automated diagnostic tools
- Highlights potential risks in medical imaging

### 2. Forensic Detection Performance
- Several state-of-the-art forensic detectors were tested on M3Dsynth
- Results:
  - Detectors trained on M3Dsynth accurately detect and localize manipulated content
  - Good performance even when training and test sets use different generation methods
  - Demonstrates strong generalization ability

## Strengths of the Study

1. **Large, Diverse Dataset**: Allows for robust testing of detection methods across various manipulation types
2. **Multiple Generation Techniques**: Enhances dataset value for assessing generalization
3. **Comprehensive Evaluation**: Thorough analysis using various metrics for both localization and detection tasks

## Limitations and Future Work

### Limitations
- Focus limited to lung CT scans and cancer nodules

### Potential Improvements
1. Expand to other types of medical images and manipulations
2. Develop new techniques specifically designed for medical image manipulation detection

## Conclusion

M3Dsynth provides a valuable resource for the research community, setting the stage for future work in the critical area of medical image forensics.
