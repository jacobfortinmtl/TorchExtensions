# Convolution as Matrix Multiplication using GEMM
The convolution operation is extremely computationally expensive both in terms of time and memory. These can be optimized by re-writing them in terms of a matrix multiplication. One such technique is im2col. Other implementations exist such as: toeplitz, Winograd, Strassen, FFT. 

Why is it done? [Characterizing and Demystifying the Implicit Convolution
Algorithm on Commercial Matrix-Multiplication Accelerators](https://cs.sjtu.edu.cn/~leng-jw/resources/Files/zhou21iiswc-im2col.pdf)  
  
_Many recent works make the assumption
of explicit im2col (image-to-column) algorithm, which lowers the convolution to a matrix multiplication via input transformation. The naive approach performs an explicit im2col
transformation to prepare the lowered feature map in the
form of the expanded matrix. **As such, this matrix can be
consumed directly by the GEMM engine without any hardware
modifications. This explicit im2col transformation leads to
significant performance and memory overheads.**_  
The authors also note that there is another way, called _implicit im2col_, which is even faster and its what used by commercial compagnies, however there is no public implementation available.

## Using im2col
Let's take a look at our example used to track the convolution through Torch's library.
Given input:
- Input: 1x3x32x32
- Kernel: 16x3x3x3

It can be noted that right before call the to sgemm_, we have the following transformed input: 
Matrix A: 1024x27
Matrix B: 27x16
Matrix C: 1024x16

The **input tensor** is transformed using the im2col technique to prepare it for matrix multiplication. This transformation arranges patches of the input tensor into a matrix (Matrix A) with shape [1024, 27], where 1024 represents the number of patches, and 27 is the size of each patch (3 channels * 3x3 kernel).  
The convolutional **kernel** is reshaped into a matrix (Matrix B) with shape [27, 16], where 27 represents the flattened kernel size for each filter, and 16 is the number of filters.  
  
Matrix Multiplication (**sgemm**): The sgemm function performs the matrix multiplication of Matrix A and Matrix B, producing an output matrix (Matrix C) with shape [1024, 16].

So what exactly does this mean?  And what is going on?  
Esentially, every sliding window of the input is re-arranged into columns, and the kernel is reshaped as well, so its compatible for multipliation. So, for the first 3x3 window, (i.e., the upper left-most patch of the input), this 2d window is taken as well as the layers beneath. Hence, we are left with 3x3 window of depth 3. In other words, patch of 27 elements. These are reshaped into a column.
There should be 1024 of these windows. That is how Matrix A is obtained.

For Matrix B, the kernel, it is simply unpooled into a vector. So, the 3x3x3 kernel becomes 27, and we have 16 of these applied -> 16x27. We now have Matrix B.
These are then multiplied together to get the output.

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

**From GPT** While it may seem that the time complexity of im2col followed by matrix multiplication should be similar to direct convolution, the practical performance benefits arise from several factors related to hardware optimization and computational efficiency. Here’s a detailed explanation:

Time Complexity
The theoretical time complexity for both im2col followed by matrix multiplication and direct convolution is indeed similar, both being O(N * M * K), where:

𝑁
N is the number of elements in the output,
𝑀
M is the size of the kernel,
𝐾
K is the number of elements in the input.
However, practical performance differs due to the following reasons:

Reasons for im2col + Matrix Multiplication Efficiency
Optimized Matrix Multiplication Libraries:

Matrix multiplication has been highly optimized in many libraries (e.g., BLAS, cuBLAS, MKL) and hardware architectures (GPUs, TPUs).
These libraries use advanced techniques like loop unrolling, cache optimization, and SIMD (Single Instruction, Multiple Data) to perform matrix multiplications much faster than what would be achievable with direct convolution implemented from scratch.
Memory Access Patterns:

im2col reorganizes the input tensor into a format that allows for sequential memory access during matrix multiplication.
Direct convolution often involves more complex memory access patterns, which can lead to inefficient use of the cache and higher latency.
Parallelization:

Matrix multiplication can be easily parallelized across multiple cores and threads, making it very efficient on modern multi-core CPUs and GPUs.
Although direct convolution can also be parallelized, the granularity of parallelism and efficiency might not be as high due to the need to handle boundary conditions and different strides/padding configurations.
Reduced Computational Overhead:

Once the im2col transformation is done, the resulting matrix multiplication can be performed in a highly optimized, continuous operation.
Direct convolution may require more conditional checks and smaller, less efficient operations spread across the entire input tensor.

An interesting reading from [this stackoverflow](https://stackoverflow.com/questions/46213531/how-is-using-im2col-operation-in-convolutional-nets-more-efficient) explains why im2col is used even if we're going over the windows already (i.e., why not just do the naive convolution).

_1. The Convolutional Layer and Fully Connected Layer are implemented using GEMM that stands for General Matrix to Matrix Multiplication.  
2. So basically in GEMM, we convert the convolution operation to a Matrix Multiplication operation by using a function called im2col() which arranges the data in a way that the convolution output can be achieved by Matrix Multiplication.  
3. Now, you may have a question instead of directly doing element wise convolution, why are we adding a step in between to arrange the data in a different way and then use GEMM.  
4. The answer to this is, scientific programmers, have spent decades optimizing code to perform large matrix to matrix multiplications, and the benefits from the very regular patterns of memory access outweigh any other losses. We have an optimized CUDA GEMM API in cuBLAS library, Intel MKL has an optimized CPU GEMM while ciBLAS's GEMM API can be used for devices supporting OpenCL.  
5. Element wise convolution performs badly because of the irregular memory accesses involved in it.  
6. In turn, Im2col() arranges the data in a way that the memory accesses are regular for Matrix Multiplication.  
7. Im2col() function adds a lot of data redundancy though, but the performance benefit of using Gemm outweigh this data redundancy.  
This is the reason for using Im2col() operation in Neural Nets._  

https://medium.com/@sundarramanp2000/different-implementations-of-the-ubiquitous-convolution-6a9269dbe77f
https://stackoverflow.com/questions/46213531/how-is-using-im2col-operation-in-convolutional-nets-more-efficient
https://cs.sjtu.edu.cn/~leng-jw/resources/Files/zhou21iiswc-im2col.pdf


