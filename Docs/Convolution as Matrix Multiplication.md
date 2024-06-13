# Convolution as Matrix Multiplication using GEMM
The convolution operation is extremely computationally expensive both in terms of time and memory. These can be optimized by re-writing them in terms of a matrix multiplication. Other implementations exist such as: Im2Col, Winograd, Strassen, FFT. 

## Using im2col
Let's take a look at our example used to track the convolution through Torch's library.
Given input:
- Input: 1x3x32x32
- Kernel: 16x3x3x3

It can be noted that right before call the to sgemm_, we have the following transformed input: 
Matrix A: 1024x27
Matrix B: 27x16
Matrix C: 1024x16

The input tensor is transformed using the im2col technique to prepare it for matrix multiplication. This transformation arranges patches of the input tensor into a matrix (Matrix A) with shape [1024, 27], where 1024 represents the number of patches, and 27 is the size of each patch (3 channels * 3x3 kernel).
The convolutional kernel is reshaped into a matrix (Matrix B) with shape [27, 16], where 27 represents the flattened kernel size for each filter, and 16 is the number of filters.
Matrix Multiplication (sgemm): The sgemm function performs the matrix multiplication of Matrix A and Matrix B, producing an output matrix (Matrix C) with shape [1024, 16].

So what exactly does this mean?  And what is going on?  
Esentially, every sliding window of the input is re-arranged into columns, and the kernel is reshaped two fit into this. So, for the first 3x3 window, (i.e., the upper left-most patch of the input), this 2d window is taken as well as the layers beneath. Hence, we are left with 3x3 window of depth 3. In other words, patch of 27 elements. These are reshaped into a column.
There should be 1024 of these windows. That is how Matrix A is obtained.

For Matrix B, the kernel, it is simply unpooled into a vector. So, the 3x3x3 kernel becomes 27, and we have 16 of these applied -> 16x27. We now have Matrix B.

### Convolution Using Matrix Multiplication (Im2Col Method) example

Suppose we have the following input tensor and kernel:

- **Input tensor (3x3)**:

  ```
  1 2 3
  4 5 6
  7 8 9
  ```

- **Kernel (2x2)**:

  ```
  1 0
  0 -1
  ```
The expected output tensor is:

```
-4 -4
-4 -4
```

#### Convolution Using Matrix Multiplication (Im2Col Method)
The first window, for instance looks like:
1. **First Window**:

   ```
   1 2
   4 5
   ```


1. **Im2Col Transformation**:

- Convert the input tensor into a matrix where each column represents a flattened sliding window.

  ```
  im2col =
  1 2 4 5
  2 3 5 6
  4 5 7 8
  5 6 8 9
  ```

2. **Flatten the Kernel**:

- Flatten the kernel into a vector.

  ```
  kernel =
  1 0 0 -1
  ```

3. **Matrix Multiplication**:

- Perform matrix multiplication between the flattened kernel and the im2col matrix.

  ```
  result = kernel * im2col
         = [1 0 0 -1] * [1 2 4 5
                         2 3 5 6
                         4 5 7 8
                         5 6 8 9]
         = [1*1 + 0*2 + 0*4 + (-1)*5,
            1*2 + 0*3 + 0*5 + (-1)*6,
            1*4 + 0*5 + 0*7 + (-1)*8,
            1*5 + 0*6 + 0*8 + (-1)*9]
         = [-4, -4, -4, -4]
  ```

4. **Reshape to Output Tensor**:

- Reshape the resulting vector back to the original output tensor shape.

  ```
  output =
  -4 -4
  -4 -4
  ```


