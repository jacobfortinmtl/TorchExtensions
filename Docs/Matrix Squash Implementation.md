# Implementing the matrix squash in Torch's backend
## Background Problem
Previously, we tried a couple of different methods to improve convolutional layers computational speeds. Let's recap the main steps.
1. Re-implementating NaN Pooling logic as closely as possible in C++, using Torch Extensions.
   - Problem: Significantly slower, do not have the same level of optimizations
2. Install Torch source and modify the backend to accomodate the changes.
   - Problem: Cannot have access to the direct convolutional implementation. It is discovered that convolutions are re-written as matrix multiplications using Im2Col technique.
3. Current attempt, re-write the input in such a way to leverage the existing optimizations already in place.
  
**Pseudo-Code for matrix Squashing**

1. For all rows of the input matrix A:
    - Check if the number of NaNs > Threshold
        - If yes: remove row, and keep index where it was removed
        - Else: continue through other rows
2. Perform convolution operation to obtain matrix C.
3. After receiving matrix C, re-insert NaNs in the indexes where the rows were deleted.

For a more in-depth exaplantion of all matrices (A, Band C) and how convolutions are implemented as matrix multiplications, see *Convolution as Matrix Multiplication*.

## Squashing the matrix (step 1)
To take full advantage of the matrix multiplication capabilities provided by the sgemm function in the BLAS library, it's crucial to ensure the input matrix, which contains the sliding windows, is compact. 
This is because these multiplication operations are optimized for dense matrices. 
Therefore, we cannot introduce gaps within the matrix, which would effectively make it sparse, as this would hinder performance. Hence, we re-create a matrix with the rows omitted.

### Using internal Torch C++ frontend.
The first technique to try was to use torch front end functions available. Our input was initialy given a contiguous memory location. So, from_blob was used to extract the given
matrice and load it into a tensor and filtered accordingly. 

```cpp
    auto A_tensor = from_blob((void*)a, {*k, *m}, at::kFloat);
    auto A_tensor_mask = (isnan(A_tensor).sum(0) < threshold);
    auto filtered_A_tensor = A_tensor.index({at::indexing::Slice(), A_tensor_mask});
```
While this could be written quickly enough, it did not allow for much customisation. Also, converting it to and from formats suitable for sgemm was not trivial, and would 
have involved playing with tensor memory, and torch internal tensor implementations. Hence this approach was discarded.

### Using vectors
A second approach was to use the vector data structure in C++ to handle going through. A vectors of vectors was created, which would point to the elements of the array. The outer vector
pointed to rows, and the innermost vector pointed to the indiviudal elements. While this made it easy to work with, it was eventually decided that it would involve too many copies of the
data (to the vector, and eventually re-written into contiguous memory). Also, vectors aren't the best in terms of speed.
```
// We will use this to remove rows with NaNs
    std::vector<std::vector<const float*>> rows;

    // setting the elements of the array to point to the specific row
    for (int i = 0; i < *m; i++) { //m = num rows in A
        std::vector<const float*> row;
        for (int j = 0; j < *k; j++) { //k = common dimension
            int index = j * *lda + i;
            row.push_back(&a[index]); 
        }
        rows.push_back(row);
    }
    std::vector<int> removed_indexes = {}; 
    // counting the number of nans in each row and removes those that have nan above threshold
    int threshold = 2;
    int row_index = 0;
    // using the erase-remove idiom to remove rows with nans
    // marking the rows to remove
    rows.erase(
      std::remove_if(
        rows.begin(),
        rows.end(),
        [&removed_indexes, &threshold, &row_index] (const auto& row) {
            int nan_count = 0;
            for (const auto& element: row) {
                if (std::isnan(*element)) {
                    nan_count++;
                }
            }
            if (nan_count > threshold) {
                removed_indexes.push_back(row_index);
                row_index++;
                return true;
            }
            row_index++;
            return false;
          }
        ),
      rows.end()
    );
```
### Using dynamic memory and c-style arrays.
Finally it was settled that the most efficient way would be to work directly with the memory itself. This was tricky for two reasons (the memory was stored in column-major
organisation in memory (so that it could be sent and used efficiently by GEMM which runs on fortran). And the second, that both the input and the output needed to be contiguous memory locations.

A first tentative approach was built by converting everything to row major organisation, and having pointers pointing to these rows, but that involved too much overhead in copying and re-orgnaising.
So, the final approach was to work directly in matrix A. The input matrix A was looped through and re-written to a new contiguous memory location.
 > Bottleneck 1: This is the first major bottleneck in speed, we need to re-create a new memory location and copy the good rows over. No solution was found to implement in place yet. 

Here is the solution for step 1:
Some notation:  
**transa** and **transb**: Characters specifying whether matrices A and B are to be transposed.'N' for no transpose, 'T' for transpose, 'C' for conjugate transpose.  
 **m** and **n**: Dimensions of the matrices. m is the number of rows of A and C, n is the number of columns of B and C.  
 **k**: Common dimension for the multiplication. If A and B are the matrices being multiplied, A has dimensions m x k and B has dimensions k x n.  
 **alpha**: Scalar multiplier for the product of matrices A and B.  
 **a** and **b**: Pointers to the matrices being multiplied.  
 **lda** and **ldb**: Leading dimensions of matrices A and B. The leading dimension is the size of the memory storage of the matrix.  
 **beta**: Scalar multiplier for matrix C.  
 **c**: Pointer to the resultant matrix after multiplication.  
 **ldc**: Leading dimension of matrix C.  
```c++
 int nan_threshold = 2; // having more NaNs than this will delete the row
    bool* row_to_remove = new bool[*m];
    int rows_removed = 0;
    int new_m = *m;

    // Identify rows to remove
    for (int i = 0; i < *m; ++i) {
        int nan_count = 0;
        row_to_remove[i] = false;
        for (int j = 0; j < *k; ++j) {
            if (std::isnan(a[j * (*lda) + i])) {
                nan_count++;
                if (nan_count > nan_threshold) {
                    row_to_remove[i] = true;
                    new_m--;
                    rows_removed++;
                    break;
                }
            }
        }
    }

    // Allocate memory for the new matrix
    float* new_a = new float[new_m * (*k)];
    int new_row = 0;

    // Write the new matrix in column-major order
    for (int j = 0; j < *k; ++j) {
        new_row = 0;
        for (int i = 0; i < *m; ++i) {
            if (!row_to_remove[i]) {
                new_a[j * new_m + new_row] = a[j * (*lda) + i];
                new_row++;
            }
        }
    }

    // Setting the pointer of a to this new memory location and updating sizes
    a = new_a;
    *m = new_m;
    *lda = new_m;
```
m, which stands for the number of rows in m and A and C, needs to be changed (in the subsequent code, m will refer to the rows of A).  
Likewise, lda is updated (which stands for leading dimension in matrix A). 
We need to update these so sgemm can work. However, we do not update the sizes of C, since we dont mind that the result of the multiplication
will be in a bigger memory location than it needs to since we will be re-inserting values in this.

When this is done, we call sgemm as follows:
```c++
  sgemm_(
        transa, transb,
        m, n, k,
        alpha,
        a, lda,
        b, ldb,
        beta,
        c, ldc);
```
This brings us to the final step.
## Re-inserting Nans
To do so, tried to methods the first was developped as follows: 
> Bottleneck 2: A second bottleneck is as follows: re-inserting into the matrix C wihtout shifting.
1. Using MEMCOPY

To do so, we will need to create a new matrix with the same dimensions as the original matrix C.
We will need to keep two pointers, one for the original matrix C and one in A, and iterate through the rows of C.
If a row was marked for removal, we will add NaNs to the row in C. Else, we will add the value that was
already in C.  
Best = Worst: O(n) time complexity and 2 copies.  
note: we are assuming that the matrix C is in column-major order, so no need calculate index
```c++
     float* new_c = new float[*ldc * *n];
     float* c_ptr = c;
     float* new_c_ptr = new_c;
     // Pointer to keep track of where we are in C, C is written back in column-major order
     for (int i = 0; i < *ldc; ++i) {
         if (row_to_remove[i]) {
             for (int j = 0; j < *n; ++j) {
                 *new_c_ptr = std::nanf("");
                 new_c_ptr++;
             }
         } else {
             for (int j = 0; j < *n; ++j) {
                 *new_c_ptr = *c_ptr;
                 new_c_ptr++;
                 c_ptr++;
             }
         }
     }
     // Copying over. Cant just change the pointer without changing a bunch of fct def
     memcpy(c, new_c, sizeof(float) * (*ldc) * (*n));
```
This wasnt the best, since the pointer for C was being passed in many different functions in Torch, one couldn't simply re-point the pointer C as this change wouldn't stay in the 
calling functions. Hence, the changes needed to be done in another memory location and copied back afterwards. This wasn't good. So another method was implemented.
2. Method 2: Right-to left in-place NaN insertions.  

To do so, we will keep two pointers in Matrix C and iterate from right to left. The first pointer will point to index *lda - 1. 
The second will point to C + *lda * *n - 1. If the value in the row is NaN, we will insert NaNs at the second pointer. Else, we will
insert at the second pointer, the value pointed by the first pointer.   
Best Case: 1 copy, O(n) time complexity  
Worst Case: 1 full copy, O(n) time complexity  
```c++
// Pointer 1: End of matrix C
float* c_ptr = c + *ldc * *n - 1;
// Pointer 2: At index *lda - 1
float* c_ptrLDA = c + *lda - 1;

// Algorithm
for (int i = *ldc - 1; i >= 0; --i){
    if (row_to_remove[i]){
        for (int j = 0; j < *n; ++j){
            *c_ptr = -1; // testing with -1 first, then im going to replace w/ std::nanf("");
            c_ptr--;
        }
    } else {
        for (int j = 0; j < *n; ++j){
            *c_ptr = *c_ptrLDA;
            c_ptr--;
            c_ptrLDA--;
        }
    }
}
```
Hence, looking at all intermediate steps, an example looks as follows (pay attention to the [17, NaN, NaN, NaN], in the 4th quadrant. The intermediate has 15 rows since 1 window/row was removed.
But after insertion, everything is at its good place and all 16 rows are accounted for (the result of the removed one is -1).
Also, to make it easier to see the differences,
instead of inserting NaNs we inserted -1:  
Padding = 0, Stride = 1. 
```
Input Tensor: 
tensor([[[[nan,  2., nan,  4.,  5.],
          [ 6., nan,  8.,  9., 10.],
          [11., 12., nan, 14., 15.],
          [16., 17., nan, 19., 20.],
          [nan, nan, nan, 24., 25.]]]])

Weight Tensor: 
tensor([[[[1., 1.],
          [1., 1.]]]])

Printing updated NaN removed matrix: 
Row 0: nan 2 6 nan 
Row 1: 2 nan nan 8 
Row 2: nan 4 8 9 
Row 3: 4 5 9 10 
Row 4: 6 nan 11 12 
Row 5: nan 8 12 nan 
Row 6: 8 9 nan 14 
Row 7: 9 10 14 15 
Row 8: 11 12 16 17 
Row 9: 12 nan 17 nan 
Row 10: nan 14 nan 19 
Row 11: 14 15 19 20 
Row 12: 16 17 nan nan 
Row 13: nan 19 nan 24 
Row 14: 19 20 24 25 

Printing after insertion 
Row 0: nan 
Row 1: nan 
Row 2: nan 
Row 3: 28 
Row 4: nan 
Row 5: nan 
Row 6: nan 
Row 7: 48 
Row 8: 56 
Row 9: nan 
Row 10: nan 
Row 11: 68 
Row 12: nan 
Row 13: -1 
Row 14: nan 
Row 15: 88 
Output after convolution: 
tensor([[[[nan, nan, nan, 28.],
          [nan, nan, nan, 48.],
          [56., nan, nan, 68.],
          [nan, -1., nan, 88.]]]])
torch.Size([1, 1, 4, 4])
```

The next steps are to parallelize the whole algorithm.
