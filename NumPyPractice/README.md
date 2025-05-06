# Comprehensive NumPy Guide

## What is NumPy?
NumPy (Numerical Python) is a powerful open-source library for numerical computing in Python. It underpins many data science tools like Pandas, SciPy, and Scikit-learn, offering efficient multidimensional arrays and a wide range of mathematical functions.

### Key Features
- **ndarray**: Fast, memory-efficient multidimensional array.
- **Vectorized Operations**: Perform computations on entire arrays without explicit loops.
- **Broadcasting**: Seamlessly operate on arrays of different shapes.
- **Linear Algebra**: Tools for matrix operations, eigenvalues, and more.

---

## Getting Started

### Installation
Install NumPy via pip:
```bash
pip install numpy
```
Import it in Python:
```python
import numpy as np
```

### Creating Arrays
NumPy’s core object is the `ndarray`. Here’s how to create them:

- **From a List**:
  ```python
  arr = np.array([1, 2, 3, 4])  # 1D array
  print(arr)  # [1 2 3 4]
  ```

- **2D Array (Matrix)**:
  ```python
  arr_2d = np.array([[1, 2], [3, 4]])  # 2x2 matrix
  print(arr_2d)
  # [[1 2]
  #  [3 4]]
  ```

- **Special Arrays**:
  - Zeros: `np.zeros((2, 3))` → 2x3 array of zeros.
  - Ones: `np.ones((3, 2))` → 3x2 array of ones.
  - Range: `np.arange(0, 10, 2)` → `[0, 2, 4, 6, 8]`.
  - Linearly Spaced: `np.linspace(0, 1, 5)` → `[0. , 0.25, 0.5 , 0.75, 1. ]`.

---

## Beginner Level

### Array Properties
- **Shape**: `arr.shape` → tuple of dimensions (e.g., `(2, 3)`).
- **Dimensions**: `arr.ndim` → number of dimensions.
- **Size**: `arr.size` → total number of elements.
- **Data Type**: `arr.dtype` → element type (e.g., `int64`).

**Example**:
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # (2, 3)
print(arr.ndim)   # 2
print(arr.size)   # 6
print(arr.dtype)  # int64
```

### Basic Operations
- **Element-wise**:
  ```python
  arr = np.array([1, 2, 3])
  print(arr + 1)  # [2 3 4]
  print(arr * 2)  # [2 4 6]
  ```

- **Matrix Operations**:
  ```python
  a = np.array([[1, 2], [3, 4]])
  b = np.array([[5, 6], [7, 8]])
  print(a + b)  # [[ 6  8]
                #  [10 12]]
  print(a @ b)  # [[19 22]
                #  [43 50]]
  ```

- **Aggregation**:
  ```python
  print(np.sum(arr))  # 6
  print(np.mean(arr)) # 2.0
  print(np.max(arr))  # 3
  ```

---

## Intermediate Level

### Indexing and Slicing
- **1D**: `arr[0]`, `arr[1:3]`.
- **2D**:
  ```python
  arr = np.array([[1, 2, 3], [4, 5, 6]])
  print(arr[0, 1])  # 2
  print(arr[1, :])  # [4 5 6]
  print(arr[:, 0])  # [1 4]
  ```

- **Boolean Indexing**:
  ```python
  print(arr[arr > 3])  # [4 5 6]
  ```

### Reshaping Arrays
```python
arr = np.arange(6)  # [0, 1, 2, 3, 4, 5]
arr_reshaped = arr.reshape(2, 3)  # 2x3 array
print(arr_reshaped)
# [[0 1 2]
#  [3 4 5]]
```

### Broadcasting
```python
arr = np.array([[1, 2], [3, 4]])
vec = np.array([1, 0])
print(arr + vec)
# [[2 2]
#  [4 4]]
```

---

## Advanced Level

### Linear Algebra
- **Inverse**: `np.linalg.inv(arr)` (square matrices only).
- **Determinant**: `np.linalg.det(arr)`.
- **Solve Equations**: `np.linalg.solve(A, b)` for `Ax = b`.

**Example**:
```python
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 11])
x = np.linalg.solve(A, b)
print(x)  # [1. 2.]
```

### Random Numbers
- `np.random.rand(2, 3)`: 2x3 array of random floats [0, 1).
- `np.random.randint(0, 10, (2, 2))`: 2x2 array of integers 0-9.
- Set seed: `np.random.seed(42)`.

### Performance Tips
- Use vectorized operations instead of loops.
- Leverage slicing for memory-efficient views.
- Specify smaller dtypes (e.g., `float32`) when possible.

---

## Real-World Example: Image Processing
Here’s how to manipulate a grayscale image as a NumPy array.

### Code
```python
import numpy as np
import matplotlib.pyplot as plt

# Create a gradient image (256x256)
image = np.linspace(0, 255, 256).repeat(256).reshape(256, 256).astype(np.uint8)

# Flip vertically
flipped_image = image[::-1, :]

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(flipped_image, cmap='gray')
axes[1].set_title('Flipped')
plt.savefig('image_processing.png')

# Statistics
print("Mean pixel value:", np.mean(image))
print("Max pixel value:", np.max(image))
```

### Explanation
- **Image Creation**: A gradient from 0 to 255 across 256 pixels, repeated for 256 rows.
- **Flip**: Slicing reverses rows vertically.
- **Output**: Saved as a PNG file with `matplotlib`.

---

## Connections
- **Pandas**: Relies on NumPy arrays for its DataFrames.
- **Visualization**: Pairs with `matplotlib` for plotting.
- **Machine Learning**: Feeds data into Scikit-learn models.

---

## Resources
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [NumPy Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)

*License: MIT*