---
title: CS229 - Lecture 2
published: 2025-07-23
description: 'Linear Regression learning algorithm'
image: ''
tags: [Machine Learning, Learning, CS229]
category: ''
draft: false 
lang: 'en'
---

## Linear Regression Overview

Linear regression is a fundamental supervised learning algorithm for regression tasks. It aims to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data.

### 1. Hypothesis Representation

The hypothesis $H(X)$ in linear regression is a linear function of the input features $X$.

-   **Single Feature:** $H(X) = \theta_0 + \theta_1X$
-   **Multiple Features:** $H(X) = \theta_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_nX_n$
-   **Vectorized Form:** $H(X) = \sum_{j=0}^{n} \theta_jX_j$ (where $X_0$ is always 1)

Here, $\theta$ (theta) are the parameters (or weights) of the model.

### 2. Cost Function (J($\theta$))

To find the best parameters $\theta$, we need to minimize a cost function. The most common one for linear regression is the **Mean Squared Error (MSE)**.

-   **Formula:** $J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (H_\theta(X^i) - Y^i)^2$
    -   $m$: Number of training examples.
    -   $H_\theta(X^i) - Y^i$: The error for the i-th training example.

The goal is to find $\theta$ that minimizes $J(\theta)$.

### 3. Gradient Descent

Gradient Descent is an iterative optimization algorithm to find the minimum of the cost function.

-   **Update Rule:** The parameters are updated in each iteration:
    $\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$
    -   $\alpha$: The learning rate, which controls the step size.
    -   $\frac{\partial J(\theta)}{\partial \theta_j}$: The partial derivative of the cost function.

-   **Derivative Term:** $\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (H_\theta(X^i) - Y^i) X_j^i$

#### Types of Gradient Descent:

1.  **Batch Gradient Descent:** Uses the entire training set to calculate the gradient in each step. It's slow for large datasets.
2.  **Stochastic Gradient Descent (SGD):** Updates parameters for each training example. It's faster but can have a noisy convergence.
3.  **Mini-Batch Gradient Descent:** A compromise, updating parameters using a small batch of training examples.

#### Code Implementation Differences

The structural difference in code between Batch and Stochastic Gradient Descent is whether the parameter update happens inside or outside the loop over the training examples.

**Batch Gradient Descent (BGD)**

In BGD, you compute the gradient for the entire dataset first and then update the parameters once.

```python
# Pseudocode for Batch Gradient Descent
repeat until convergence {
  // Calculate gradients for all examples
  error_sum = 0
  for i = 1 to m {
    error = h(x_i) - y_i
    error_sum += error * x_i
  }
  
  gradient = (1/m) * error_sum
  
  // Update parameters after summing over all examples
  theta = theta - alpha * gradient
}
```

**Stochastic Gradient Descent (SGD)**

In SGD, you update the parameters for *each* training example, one by one.

```python
# Pseudocode for Stochastic Gradient Descent
repeat until convergence {
  // Loop through each example in the dataset
  for i = 1 to m {
    // Calculate gradient for a single example
    error = h(x_i) - y_i
    gradient = error * x_i
    
    // Update parameters for each example
    theta = theta - alpha * gradient
  }
}
```

### 4. Normal Equation

For linear regression, there is a direct, non-iterative method to solve for the optimal $\theta$.

-   **Formula:** $\theta = (X^TX)^{-1}X^TY$
    -   $X$: The design matrix containing all training examples' features.
    -   $Y$: The vector of target values.

-   **Trade-offs:**
    -   **Advantage:** No need to choose a learning rate $\alpha$, and it's fast for small datasets.
    -   **Disadvantage:** Calculating the inverse of $X^TX$ is computationally expensive ($O(n^3)$), making it slow for a large number of features $n$.
