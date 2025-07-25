---
title: CS229-Lecture1
published: 2025-07-19
description: 'Introduction to Machine Learning'
image: ''
tags: [Machine Learning, Learning, CS229]
category: ''
draft: false 
lang: 'en'
---

# Introduction to Machine Learning

## Definition

- **Arthur Samuel**: "Field of study that gives computers the ability to learn without being explicitly programmed"
- **Tom Mitchell**: "A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E"

## Categories

### Supervised Learning
The most widely used tool, involving learning a mapping from inputs (X) to outputs (Y) given a dataset where both X and Y are provided during training.

- **Regression**: Predicting a continuous output (e.g., housing prices)
- **Classification**: Predicting a discrete output (e.g., classifying a tumor)

### Unsupervised Learning
Finding patterns in data without given labels (only inputs X, no Y).

### Reinforcement Learning
Training an agent to make decisions by providing reward signals for desired behaviors.

# AI / ML / DL

## Artificial Intelligence

AI is the overarching field dedicated to creating machines or systems that can *mimic human intelligence*. The goal is for these machines to perform tasks that traditionally require human cognitive abilities, such as:
- **Learning**: Acquiring knowledge and skills from experience.
- **Reasoning**: Drawing conclusions from information.
- **Problem-solving**: Finding solutions to complex issues.
- **Perception**: Understanding and interpreting sensory information (like vision and hearing).
- **Decision-making**: Choosing the best course of action.
- **Understanding natural language**: Processing and responding to human language.

## Machine Learning / Deep Learning

| Aspect | Machine Learning | Deep Learning |
|--------|------------------|---------------|
| **Definition** | Algorithms that learn patterns from data without explicit programming | Specialized subset of machine learning that uses artificial neural networks with multiple hidden layers |
| **Data Requirements** | Can work with smaller datasets | Requires large amounts of data |
| **Feature Engineering** | Manual feature extraction often required | Automatic feature extraction |
| **Computational Power** | Lower computational requirements | High computational power needed (GPUs) |
| **Architecture** | Encompasses a wide range of algorithms:<br>• **Linear Regression**: Fits a straight line to data<br>• **Decision Trees**: Creates tree-like decision models<br>• **SVMs**: Finds optimal separation hyperplanes<br>• **KNN**: Classifies based on nearest neighbors | Built upon Artificial Neural Networks (ANNs):<br>• Multiple hidden layers<br>• Each layer processes input from previous layer<br>• Extracts increasingly complex features<br>• Inspired by human brain structure |
| **Interpretability** | More interpretable models | Often "black box" - less interpretable |
| **Performance** | Good for structured data and smaller problems | Excels with unstructured data (images, text, audio) |
