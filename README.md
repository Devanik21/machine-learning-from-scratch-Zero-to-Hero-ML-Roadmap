# 🌟 Machine Learning Projects: From Fundamentals to Advanced Models 🌟

Welcome to the comprehensive machine learning repository! This project serves as an organized collection of Jupyter Notebooks, illustrating various fundamental and advanced machine learning algorithms. We emphasize a deep mathematical understanding and practical implementation using industry-standard libraries such as `scikit-learn` and `statsmodels`.

We kindly invite you to explore the directories, each dedicated to specific families of models. Let us delve into the beautiful mathematics that govern these learning algorithms! 🚀

---

## 📈 1. Linear Regression Models

Linear regression aims to model the relationship between a dependent variable $y$ and one or more independent variables $X$.

### Simple & Multiple Linear Regression

The hypothesis function is defined as:
$$ h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n = \theta^T x $$

The objective is to find the parameters $\theta$ that minimize the **Ordinary Least Squares (OLS)** cost function, defined as the Mean Squared Error (MSE):
$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 $$

This convex optimization problem has a beautiful closed-form solution via the Normal Equation:
$$ \theta = (X^T X)^{-1} X^T y $$

### Polynomial Linear Regression

When the data relationship is highly non-linear, we apply polynomial transformations to the features:
$$ h_\theta(x) = \theta_0 + \theta_1 x + \theta_2 x^2 + \dots + \theta_d x^d $$
While the features are mapped to a higher-dimensional polynomial space, the model remains *linear* with respect to the parameters $\theta$, allowing us to solve it using the standard OLS framework. 🧮

---

## ⚖️ 2. Regularized Linear Models

To combat overfitting in high-dimensional spaces, we enforce penalty terms on the magnitude of the parameters.

### Ridge Regression ($L_2$ Penalty)

Ridge regression introduces an $\ell_2$-norm regularization term to the OLS cost function:
$$ J(\theta) = \text{MSE}(\theta) + \alpha \frac{1}{2} \sum_{i=1}^{n} \theta_i^2 $$
The hyperparameter $\alpha$ controls the regularization strength. This modifies the closed-form Normal Equation to:
$$ \theta = (X^T X + \alpha A)^{-1} X^T y $$
where $A$ is the identity matrix with the top-left element set to $0$ (so we do not penalize the bias term $\theta_0$).

### LASSO Regression ($L_1$ Penalty)

Least Absolute Shrinkage and Selection Operator (LASSO) penalizes the $\ell_1$-norm of the coefficients, effectively performing automated feature selection by driving irrelevant parameter weights exactly to zero:
$$ J(\theta) = \text{MSE}(\theta) + \alpha \sum_{i=1}^{n} |\theta_i| $$
Because the $\ell_1$-norm is not differentiable at $0$, we typically rely on subgradient methods or coordinate descent algorithms.

### ElasticNet Regression

ElasticNet elegantly combines the continuous differentiability of Ridge with the sparsity-inducing properties of LASSO. It minimizes:
$$ J(\theta) = \text{MSE}(\theta) + r \alpha \sum_{i=1}^{n} |\theta_i| + \frac{1 - r}{2} \alpha \sum_{i=1}^{n} \theta_i^2 $$
where $r \in [0, 1]$ represents the mix ratio between the $L_1$ and $L_2$ penalties. 📐

---

## 🔬 3. Logistic Regression & Classification

Logistic Regression estimates the probability that an instance belongs to a specific binary class. It passes the linear hypothesis $z = \theta^T x$ through the **Sigmoid (Logistic) function**:
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
The output bounds between $0$ and $1$, yielding the predicted probability $\hat{p} = \sigma(\theta^T x)$.

The objective is to minimize the Log-Loss (Cross-Entropy) function, a convex cost function:
$$ J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right] $$
We typically use Gradient Descent or advanced optimization routines (like L-BFGS) to compute the optimal weight vector $\theta$. 📉

---

## 🗡️ 4. Support Vector Machines (SVM)

SVM is a powerful algorithm capable of performing linear or non-linear classification by defining an optimal margin separating hyperplane.

### Primal Form & Margin Maximization

The objective of a soft-margin SVM is to find a hyperplane $w^T x + b = 0$ that maximizes the distance between the two classes while allowing some margin violations (controlled by the hyperparameter $C$).
$$ \min_{w, b, \zeta} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{m} \zeta_i $$
Subject to the constraints: $y^{(i)}(w^T x^{(i)} + b) \ge 1 - \zeta_i \quad \text{and} \quad \zeta_i \ge 0$.

### Dual Form & The Kernel Trick

By computing the Lagrangian Dual of the constrained optimization problem, the formulation depends only on the dot product of the input feature vectors:
$$ \max_{\alpha} \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i, j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} K(x^{(i)}, x^{(j)}) $$
Where $\alpha_i$ are the Lagrangian multipliers and $K(x, x') = \phi(x)^T \phi(x')$ is a Kernel function.
Common kernels include the **Radial Basis Function (RBF)**, allowing us to compute decision boundaries in an implicit, infinite-dimensional feature space. 🌌

---

## 🧮 5. Naive Bayes Classification

Naive Bayes is a probabilistic classifier based on Bayes' Theorem. It rests on the "naive" assumption of conditional independence between every pair of features, given the class variable.

Given a feature vector $X = (x_1, \dots, x_n)$ and a class variable $y$, Bayes' Theorem states:
$$ P(y \mid x_1, \dots, x_n) = \frac{P(y) P(x_1, \dots, x_n \mid y)}{P(x_1, \dots, x_n)} $$
Assuming conditional independence, $P(x_1, \dots, x_n \mid y) = \prod_{i=1}^{n} P(x_i \mid y)$.
The Maximum A Posteriori (MAP) estimation calculates the most probable class:
$$ \hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i \mid y) $$
Common variations like Gaussian and Multinomial Naive Bayes model the conditional distributions differently. 🔔

---

## 🌲 6. Decision Trees & Random Forests

Decision trees split the input space into a set of rectangular regions. At each node, a feature $k$ and a threshold $t_k$ are chosen to maximize the Information Gain.

### Impurity Metrics

The CART algorithm uses the **Gini Impurity** to measure node purity:
$$ G_i = 1 - \sum_{k=1}^{n} p_{i,k}^2 $$
Or alternatively, **Entropy**:
$$ H_i = - \sum_{k=1}^{n} p_{i,k} \log_2(p_{i,k}) $$
Where $p_{i,k}$ is the ratio of class $k$ instances in the $i$-th node. The tree recursively splits nodes until maximum depth or a minimum samples constraint is met.

### Ensemble Learning: Random Forests

Random Forests utilize **Bagging (Bootstrap Aggregating)**. Multiple decision trees are trained on subsets of the training data sampled with replacement. Additionally, each split only considers a random subset of features $\sqrt{n}$. This reduces variance and mitigates the overfitting tendency of singular deep decision trees.
$$ \hat{f}_{\text{rf}}(x) = \frac{1}{B} \sum_{b=1}^{B} f_b(x) $$
Where $B$ is the total number of trees constructed. 🌳

---

## 🧠 7. Artificial Neural Networks (ANN)

Artificial Neural Networks form the basis of Deep Learning. They consist of highly interconnected processing units (neurons), structured in an input layer, hidden layers, and an output layer.

### Forward Propagation

Data $X$ flows forward through the network. At each hidden layer $l$, a linear transformation is applied, followed by a non-linear activation function $g$:
$$ Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]} $$
$$ A^{[l]} = g(Z^{[l]}) $$
Where $W^{[l]}$ is the weight matrix and $b^{[l]}$ is the bias vector. $A^{[0]} = X$. Common activation functions include the **Sigmoid** $\sigma(z)$, **Hyperbolic Tangent** $\tanh(z)$, and the **Rectified Linear Unit (ReLU)** $\max(0, z)$.

### Backpropagation & Optimization

The network's parameters are updated via backpropagation, an application of the chain rule from calculus. We compute the gradient of the loss function $L(y, \hat{y})$ with respect to the weights:
$$ \frac{\partial L}{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} A^{[l-1]T} $$
$$ \frac{\partial L}{\partial b^{[l]}} = \frac{1}{m} \sum dZ^{[l]} $$
Where $dZ^{[l]} = dA^{[l]} * g'(Z^{[l]})$.

Optimization algorithms like Stochastic Gradient Descent (SGD) or Adam iteratively adjust the parameters:
$$ W^{[l]} = W^{[l]} - \alpha \frac{\partial L}{\partial W^{[l]}} $$
Where $\alpha$ represents the learning rate. 🌐

---

We sincerely hope this repository provides a rigorous and clear perspective on machine learning algorithms. Please feel free to delve into the code and explore the practical implementations! ✨
