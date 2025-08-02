---
title: CS229 - Lecture 3
published: 2025-07-29
description: 'Locally Weighted & Logistic Regression'
image: ''
tags: [Machine Learning, Learning, CS229]
category: ''
draft: false 
lang: 'en'
---

<iframe width="560" height="315" src="https://www.youtube.com/embed/het9HFqo1TQ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


## Key Terminology

- **Parametric learning algorithm**: Fits a fixed set of parameters $\theta_i$ to data. Once trained, the model complexity remains constant regardless of training set size.
- **Non-parametric learning algorithm**: The amount of data/parameters needed grows with the size of the training data. Model complexity scales with dataset size.

# Locally Weighted Regression

## Linear Regression Recap
In standard linear regression, we fit parameters $\theta$ to minimize the cost function:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\theta^T x^{(i)} - y^{(i)})^2$$

The prediction for a new input $x$ is: $h_\theta(x) = \theta^T x$

## Locally Weighted Regression (LWR)
Instead of fitting one global model, LWR fits a local model for each prediction by minimizing:

$$J(\theta) = \sum_{i=1}^{m} w^{(i)} (\theta^T x^{(i)} - y^{(i)})^2$$

where $w^{(i)}$ is a weight function that gives higher weight to training examples close to the query point.

### Weight Function
The most common weight function is the Gaussian kernel:

$$w^{(i)} = \exp\left(-\frac{(x^{(i)} - x)^2}{2\tau^2}\right)$$

where:
- $\tau$ is the **bandwidth parameter** that controls how quickly the weight decreases with distance
- If $|x^{(i)} - x|$ is small, then $w^{(i)} \approx 1$ (high weight)
- If $|x^{(i)} - x|$ is large, then $w^{(i)} \approx 0$ (low weight)

### Key Properties
- **Non-linear Regression**: Models complex, non-linear relationships in data
- **Non-parametric**: No fixed set of parameters to learn
- **Lazy learning**: Computation happens at prediction time, not training time
- **Local fitting**: Each prediction uses a locally weighted subset of training data
- **Bandwidth sensitivity**: 
  - Small $\tau$: More local, higher variance, lower bias
  - Large $\tau$: More global, lower variance, higher bias

# Why least squares?

When faced with a regression problem, why might linear regression, and specifically why might the least-squares cost function J, be a reasonable choice?

TL;DR: watch these to understand the math
[Probability vs Likelihood](https://www.youtube.com/watch?v=pYxNSUDSFH4)
[Maximum Likelihood](https://www.youtube.com/watch?v=XepXtl9YKwc)
[Maximum Likelihood For the Normal Distribution](https://www.youtube.com/watch?v=Dn6b9fCIUpM)

Let's assume that the target variable and the inputs are related via the equation:
$$y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$$

where $\epsilon^{(i)}$ is an error term or random noise. We also assume that the $\epsilon^{(i)}$ are distributed IID (independently and identically distributed) according to a Gaussian distribution:
$$\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$$

## Maximum Likelihood Derivation

The density of $\epsilon^{(i)}$ is given by:
$$p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(\epsilon^{(i)})^2}{2\sigma^2}\right)$$

Since $y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$, we have $\epsilon^{(i)} = y^{(i)} - \theta^T x^{(i)}$.

Therefore, the probability density of $y^{(i)}$ given $x^{(i)}$ and $\theta$ is:
$$p(y^{(i)} | x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right)$$

This means: $y^{(i)} | x^{(i)}; \theta \sim \mathcal{N}(\theta^T x^{(i)}, \sigma^2)$

## Likelihood Function

Given a training set $\{(x^{(i)}, y^{(i)}); i = 1, \ldots, m\}$, the likelihood of the parameters is:
$$L(\theta) = \prod_{i=1}^{m} p(y^{(i)} | x^{(i)}; \theta)$$

$$L(\theta) = \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right)$$

## Log-Likelihood

Taking the logarithm (which is monotonic, so maximizing $L(\theta)$ is equivalent to maximizing $\log L(\theta)$):

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^{m} \log \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right)$$

$$\ell(\theta) = m \log \frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{2\sigma^2} \sum_{i=1}^{m} (y^{(i)} - \theta^T x^{(i)})^2$$

## Maximum Likelihood Estimation

To maximize $\ell(\theta)$, we need to minimize:
$$\sum_{i=1}^{m} (y^{(i)} - \theta^T x^{(i)})^2$$

This is exactly the least-squares cost function! Therefore, **under the assumption that errors are Gaussian, maximum likelihood estimation leads directly to the least-squares objective**.

# Logistic Regression

For classification problems, we want $h_\theta(x) \in [0, 1]$ to represent probabilities.

We define the **hypothesis function**:
$$h_\theta(x) = g(\theta^T x) = \frac{1}{1+e^{-\theta^T x}}$$

where $g(z) = \frac{1}{1+e^{-z}}$ is called the **logistic function** or **sigmoid function**.

## Properties of the Sigmoid Function

### Derivative of the Sigmoid Function
$$g'(z) = \frac{d}{dz} \frac{1}{1+e^{-z}}$$

Using the chain rule:
$$g'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}}$$

$$= \frac{1}{1+e^{-z}} \cdot \frac{1+e^{-z}-1}{1+e^{-z}} = g(z)(1-g(z))$$

### Key Properties
- $g(z) \in (0,1)$ for all $z \in \mathbb{R}$
- $g(0) = 0.5$
- $\lim_{z \to \infty} g(z) = 1$ and $\lim_{z \to -\infty} g(z) = 0$
- $g(-z) = 1 - g(z)$ (symmetry)

## Probabilistic Interpretation

We model:
- $P(y = 1 | x; \theta) = h_\theta(x) = g(\theta^T x)$
- $P(y = 0 | x; \theta) = 1 - h_\theta(x) = 1 - g(\theta^T x)$

This can be written compactly as:
$$P(y | x; \theta) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y}$$

## Maximum Likelihood Estimation

Given training set $\{(x^{(i)}, y^{(i)}); i = 1, \ldots, m\}$ where $y^{(i)} \in \{0,1\}$:

### Likelihood Function
$$L(\theta) = \prod_{i=1}^{m} P(y^{(i)} | x^{(i)}; \theta) = \prod_{i=1}^{m} (h_\theta(x^{(i)}))^{y^{(i)}} (1 - h_\theta(x^{(i)}))^{1-y^{(i)}}$$

### Log-Likelihood
$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^{m} \left[ y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)})) \right]$$

## Gradient Ascent

Since we want to **maximize** the log-likelihood $\ell(\theta)$, we use gradient ascent.

### Computing the Gradient
The partial derivative of the log-likelihood with respect to $\theta_j$ is:
$$\frac{\partial}{\partial \theta_j} \ell(\theta) = \sum_{i=1}^{m} \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}$$

**Derivation:**
$$\frac{\partial}{\partial \theta_j} \ell(\theta) = \sum_{i=1}^{m} \left[ y^{(i)} \frac{1}{h_\theta(x^{(i)})} \frac{\partial h_\theta(x^{(i)})}{\partial \theta_j} + (1-y^{(i)}) \frac{1}{1-h_\theta(x^{(i)})} \frac{\partial}{\partial \theta_j}(1-h_\theta(x^{(i)})) \right]$$

Since $\frac{\partial h_\theta(x^{(i)})}{\partial \theta_j} = h_\theta(x^{(i)})(1-h_\theta(x^{(i)})) x_j^{(i)}$:

$$= \sum_{i=1}^{m} \left[ y^{(i)} (1-h_\theta(x^{(i)})) - (1-y^{(i)}) h_\theta(x^{(i)}) \right] x_j^{(i)}$$

$$= \sum_{i=1}^{m} \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}$$

### Gradient Ascent Update
To maximize $\ell(\theta)$, we move in the direction of the gradient:
$$\theta_j := \theta_j + \alpha \frac{\partial}{\partial \theta_j} \ell(\theta) = \theta_j + \alpha \sum_{i=1}^{m} \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}$$

### Alternative: Minimize Negative Log-Likelihood
Equivalently, we can **minimize** the negative log-likelihood (cost function):
$$J(\theta) = -\ell(\theta) = -\sum_{i=1}^{m} \left[ y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)})) \right]$$

Then the gradient descent update becomes:
$$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j} = \theta_j - \alpha \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

Both approaches are equivalent and lead to the same solution!

## Newton's Method

Newton's method provides faster convergence by using second-order information.

### Newton's Update Rule
$$\theta := \theta - H^{-1} \nabla_\theta J(\theta)$$

where $H$ is the **Hessian matrix** of second derivatives.

### Computing the Hessian
For logistic regression, the Hessian is:
$$H = \frac{1}{m} \sum_{i=1}^{m} h_\theta(x^{(i)})(1-h_\theta(x^{(i)})) x^{(i)} (x^{(i)})^T$$

In matrix form:
$$H = \frac{1}{m} X^T S X$$

where $S$ is a diagonal matrix with $S_{ii} = h_\theta(x^{(i)})(1-h_\theta(x^{(i)}))$.

### Newton's Method Algorithm
1. Initialize $\theta$ (e.g., $\theta = 0$)
2. Repeat until convergence:
   - Compute gradient: $g = \frac{1}{m} X^T (h - y)$ where $h$ is vector of predictions
   - Compute Hessian: $H = \frac{1}{m} X^T S X$
   - Update: $\theta := \theta - H^{-1} g$

### Advantages and Disadvantages

**Advantages:**
- **Quadratic convergence**: Much faster than gradient descent near optimum
- **No learning rate**: Automatically determines step size
- **Fewer iterations**: Typically converges in 5-10 iterations

**Disadvantages:**
- **Computational cost**: $O(n^3)$ per iteration due to matrix inversion
- **Memory requirements**: Must store and invert $n \times n$ Hessian matrix
- **Numerical stability**: Hessian might be singular or ill-conditioned

### When to Use Newton's Method
- **Small to medium features** ($n \lesssim 10,000$)
- **Well-conditioned problems**
- **When fast convergence is critical**


