
# Sample ML/AI Projects

## Overview
The following projects are samples of transfer learning using Transformer models, using a variety of tasks and data sources in multiple contexts and modalities. 
- Project 1, [AI Text Classification](#ai-text-classification), uses the GPT-2 Large Language Model to identify AI generated text. 
- Project 2, [Fake Face Detection](#fake-face-detection), uses an Image Transformer to identify AI generated photos. 

See [environment setup](#environment-setup) for environmental setup.

All samples developed by [Daniel Kaiser](https://www.linkedin.com/in/daniel-kaiser-a397a419b/), PhD.

---

## AI Text Classification

**Model:** GPT-2

**Code:** [ai_text_detection](https://github.com/dpk2001/sample_projects/tree/main/ai_text_detection)


This task implements a classifier model to identify AI generated text by using a dataset with both Human created and AI generated texts.

The code sample uses the base of a pre-trained GPT-2 Large Language Model, with an added linear head for use in classification.

**To Run:**

First, run `text_real_vs_fake_make_data.py` to generate GPT processed last hidden states for subsequent processing by classifier in `text_real_vs_fake.ipynb`.

### Results:

Naive accuracy (taking most common label as prediction): 74%

:white_check_mark: **Achieved accuracy: 90%** 

---


## Fake Face Detection
**Model:** DeiT

**Code:** [fake_face_detection](https://github.com/dpk2001/sample_projects/tree/main/fake_face_detection)

This takes implements a classifier model to identify AI generated faces by using a dataset with both real and AI generated faces.

The code sample uses the base of (two different sizes of) a pre-trained DeiT Image Transformer model, with an added linear head for use in classification.

**To Run:**

All the code is contained in `face_real_vs_fake_small.ipynb` and `face_real_vs_fake_tiny.ipynb`, which differ in the size of the model used.

### Results:

Naive accuracy (taking most common label as prediction): 51%

Achieved accuracy ("tiny" model): 95%

:white_check_mark: **Achieved accuracy ("small" model): 97%**

---

## Environment setup
To run the code samples in this repository, create a Python 3.11 virtual environment and pip install the `requirements_py311.txt` file included.
