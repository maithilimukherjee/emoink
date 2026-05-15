# Emoink

An affective handwriting analysis system using feature engineering and probabilistic sequence modelling.

## Overview

Emoink is a machine learning system designed to analyze handwriting patterns and extract emotional indicators through advanced feature engineering and Hidden Markov Model (HMM) based sequence analysis. The project combines computer vision techniques with probabilistic graphical models to recognize emotional states based on handwriting characteristics.

## Features

- **Feature Engineering**: Extract relevant handwriting features for emotional analysis
- **Probabilistic Sequence Modelling**: Utilize HMM and Viterbi algorithm for pattern recognition
- **Pattern Recognition**: Identify emotional states from handwriting samples
- **Machine Learning Pipeline**: End-to-end ML system for affective computing

## Technologies

- **Language**: Python
- **Key Libraries**:
  - NumPy and SciPy for numerical computations
  - OpenCV and scikit-image for image processing
  - Matplotlib for visualization

## Project Structure

```
emoink/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── main.py
│   ├── preprocessing/
│   │   ├── grayscale.py
│   │   ├── denoise.py
│   │   ├── threshold.py
│   │   ├── deskew.py
│   │   └── segment.py
│   ├── features/
│   │   └── extractfeats.py
│   ├── inference/
│   │   ├── affective_state.py
│   │   └── normalizescores.py
│   ├── hmm/
│   │   ├── hmmmodel.py
│   │   └── viterbi.py
│   └── utils/
│       ├── imageloader.py
│       └── visualizer.py
└── README.md
```

## Tech Stack

- Python  
- OpenCV  
- NumPy  
- Hidden Markov Models (HMM)  
- Viterbi Algorithm  
- Classical Computer Vision techniques  

---

## Key Concepts Used

- Image preprocessing pipeline design  
- Feature engineering from visual signals  
- Probabilistic classification  
- Sequence modeling using HMM  
- Temporal decoding using Viterbi algorithm  

---

## Limitations

- Small dataset (prototype scale only)  
- Heuristic-based inference layer  
- Emotion labels are not clinically validated  
- Performance depends on handwriting quality and image clarity  

---

## Future Improvements

- Replace heuristic inference with trained ML model  
- Build labeled handwriting emotion dataset  
- Improve segmentation robustness  
- Add CNN-based feature extraction  
- Learn HMM parameters from data instead of manual setup  

---

## Purpose

This project is intended for:

- Affective computing exploration  
- Computer vision + ML integration practice  
- Academic submission / portfolio showcase  
- Understanding probabilistic sequence modeling in real-world data  
