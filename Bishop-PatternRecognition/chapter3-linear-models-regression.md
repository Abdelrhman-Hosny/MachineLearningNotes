[toc]

# <u>**Chapter 3 - Linear Models (Regression)**</u>

- The goal of regression is to **predict 1 or more continuous *target* variables $t$ given the value of a $D$-dimensional vector $\mathbf{x}$  of *input* variables**.
- The polynomial curve fitting is a **specific example**  of a **broad class of function*s* called *linear regression models*** 
  - The models are called **linear** as they are linear functions **of the *adjustable parameters* **(can be **non-linear** functions of **inputs**)
- The simplest form of **linear regression models** are **linear functions of the input variables**.
  - However, we can obtain a **more useful** class of functions by taking **linear combinations of a fixed set of *non-linear functions of the input variables***, these functions are known as **basis functions**.
    - These models are **linear functions of the parameters**, which gives them **simple analytical properties**.
    - They can be **non-linear** w.r.t the **input variables**.
- Linear models have significant **limitations** as **practical techniques** in ML, particularly in problems involving **high dimensional input spaces**.
  - They do however have **nice analytical properties** and **form the foundation** for more sophisticated models

****

## **<u>Linear Basis Function Models</u>**

- The simplest linear model for regression is just a **linear combination** of the **input variables**
  $$
  y(\mathbf{x,w}) = w_0 + w_1x_1+....+w_Dx_D
  $$
  having the function be **linear** in the **input variables** imposes **significant limitations**.

  - So we extend the model by considering **non linear combinations** of **fixed <u>non-linear</u> functions of the <u>input variables</u>**
    $$
    y(\mathbf{x,w}) = w_0 + \sum^{M-1}_{j=1} w_j \phi_j(\mathbf{x})
    $$
    where $\phi(\mathbf{x})$ are known as **basis functions**.

  - The total number of parameters in this model is $M$.

- For convenience, we often add a new value $\phi_0(\mathbf{x}) = 1$, this allows us to express $y(\mathbf{x,w})$ as follows
  $$
  y(\mathbf{x,w}) = \sum^{M-1}_{j=0} w_j \phi_j(\mathbf{x}) = \mathbf{w}^T\phi(\mathbf{x})
  $$
  where $\mathbf{w} = (w_0, w_1,...,w_{M-1})^T$ and $\phi=(\phi_0,....,\phi_{M-1})^T$.

- In many applications of pattern recognition, we will apply some form of **fixed pre-processing** or **feature extraction** to the original data variables $\mathbf x$, we can represent these operations in terms of the **basis functions** $\{ \phi_j(\mathbf x )\}$

****

### **<u>Examples of Basis Functions</u>**

- In the polynomial curve fitting example, we used the basis function $\phi_j(x) = x^j$.

  - A limitation for these functions is that they are **global functions**, so changes in one region of the input space affects all other regions.
    - This can be fixed by dividing the input space into regions and fitting a **different polynomial** for each region. functions that do this are called **spline functions**

- **<u>Gaussian Basis functions</u>**
  $$
  \phi_j(\mathbf x) = \exp\{ -\frac{||\mathbf x-\mu_j||^2}{2s^2}\}
  $$

  - where $\mu_j$ can be obtained through some sort of reductions. (using k-means to find the centers that represent the data, and in that case $\phi$ would be $k$-dimensional)
  - The parameter $s$ governs the **spatial scale**.
  - Notice that $2s^2$ is just a scaling factor which is not that important, as the $\mathbf w$ are adaptive and will decrease/increase to match the target.

- **<u>Sigmoidal basis function</u>**
  $$
  \phi_j(\mathbf x) = \sigma(\frac{\mathbf x - \mu_j}{s})
  $$
  where $\sigma$ is the sigmoid function

  - we can replace the sigmoid with $\tanh$ as they are closely related $\tanh(a) = 2\sigma(a) - 1$.

- **<u>Fourier basis function</u>**

  - Each basis function represents a **specific frequency** and has **infinite spatial extent**.
  - In many signal processing applications, it it of interest to consider functions that are **localized in both space and frequency**, leading to a class of functions called **wavelets**. 
  - **wavelets** are designed to be **mutually orthogonal** to simplify their application.
  - Wavelets are **most applicable** when the input values live on a **regular lattice**, such as **successive time points in a temporal sequence**, or **pixels in an image**.

![](./Images/ch3/ex-basis-functions.png)

****

The analysis applied in this chapter is usually **independent** of the **choice of basis functions**.

****

## <u>**ML and Least Squares**</u>

- Previously, we fitted polynomial functions to datasets by **minimizing a sum-of-squares** error function.

  - We showed that the error function could be motivated as the ML solution under a **Gaussian noise model**.

- We assume $t$ is given by
  $$
  t = y(\mathbf{x,w}) + \epsilon
  $$
  where $\epsilon$ is a $\mathcal{N}(0,\beta^{-1})$, which allows us to say
  $$
  p(t|\mathbf{x,w},\beta) = \mathcal{N}(t|y(\mathbf{x,w}), \beta^{-1})
  $$
   if we assume a **squared loss function**, the **optimal prediction** for a new value of $\mathbf x$ will be given by **the conditional mean of the target variable**, in case of a Gaussian distribution, the conditional mean will be
  $$
  E[t|\mathbf x] = \int t * p(t|x) dt = y(\mathbf{x, w})
  $$

- Note that **Gaussian noise assumption** implies that the $p(t|\mathbf{x})$ is **unimodal**, which may be **inappropriate for some applications**.

- If we consider the likelihood function
  $$
  p(\mathbf{t|X,w}\beta) = \prod^N_{n=1} \mathcal{N}(t_n|\mathbf w^T\phi(\mathbf x_n), \beta^{-1})
  $$
  Since in regression problem, we don't seek to model $\mathbf X$, we will drop the $\mathbf x$ from expressions like $p(\mathbf{ t| x, w}, \beta)$ to keep the notation **uncluttered**

  - We will take the log likelihood which is
    $$
    \ln p(\mathbf t|\mathbf w, \beta) = \sum^N_{n=1} \ln \mathcal{N}(t_n|\mathbf w^T \phi(\mathbf x_n), \beta^{-1})
    \\ = \frac{N}2 \ln \beta - \frac{N}2 \ln2\pi - \beta E_D(\mathbf w)
    $$
    where $E_D(\mathbf w) = \frac{1}2 \sum^N_{n = 1} \{ t_n - \mathbf w^T \phi(\mathbf x_n)\}^2$.

  - We can now compute the maximum likelihood for $\mathbf w$ and $\beta$
    $$
    \nabla \ln p(\mathbf{ t|w},\beta) = \sum^N_{n=1} \{t_n - \mathbf w^T \phi(\mathbf x_n) \} \phi(\mathbf x_n)^T
    $$
    if we set the gradient to zero and solve, we obtain
    $$
    \mathbf w_{ML} = (\phi^T\phi)^{-1} \phi^T \mathbf t
    $$
    which is know as the **normal equation** for the **least squares problem**.

    $\phi$ is an $N \times M$ matrix called the **design matrix** whose elements are given by $\phi_{nj} =\phi_j(\mathbf x_n)$

  - The quantity $\phi^\dagger = (\phi^T \phi)^{-1} \phi^T$ which is known as **Moore-Penrose pseudo-inverse**

    It can be regarded as a **generalization** of the **inverse** to **non square matrices**.

    - If the matrix is square and invertible $A^\dagger = A^{-1}$.

- We can gain some insight into the **role of the bias parameter** $w_0$.
  $$
  E_D(\mathbf w) = \frac{1}2 \sum^N_{n=1} \{ t_n - w_0 - \sum^{M-1}_{j=1} w_j \phi_j(\mathbf x_n)\}^2
  $$
  setting the derivative w.r.t $w_0$ to zero, and solving for $w_0$, we get
  $$
  w_0 = \bar t- \sum^{M-1}_{j=1} w_j \bar \phi_j
  $$
  where 
  $$
  \bar t = \frac{1}N \sum^N_{n=1}t_n \ \& \ \bar \phi_j = \frac{1}N \sum^N_{n=1} \phi_j(\mathbf x_n)
  $$
  thus the bias $w_0$ **compensates** for the **difference** between **averages of the target variable** and the **weighted sum of the average of the basis function values**. (on the training set)

- We can also maximize the log likelihood w.r.t $\beta$ to get
  $$
  \frac{1}{\beta_{ML}} = \frac{1}N \sum^N_{n=1} \{ t_n - \mathbf w^T_{ML} \phi(\mathbf x_n) \}^2
  $$
  so it is given by the **residual variance** of the **target values** around the **regression function**.

****

## **<u>Geometry of least squares</u>**