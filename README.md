Full project work inside main.ipynb
# COMP 490 Project Proposal

## Implementing NaN Pooling in PyTorch's C/C++ Framework

## Background:

Convolutional Neural Networks (CNNs) are fundamental to brain imaging. They function by applying a series of convolutional operations to an image.

CNNs apply this filtering operation repeatedly in varying sizes and formats, allowing the model to gain an *understanding* of the image. However, the complexity of the model increases substantially with the addition of more filters, leading to intensive memory and computational demands due to the vast complexity and numerous parameters [1].

A layer known as Max Pooling is incorporated into these networks. This layer maintains the same depth of the model while reducing its spatial size, thereby enhancing computational efficiency. It also preserves the key features of an image [2]. Several other layers are similarly employed to manage computation by adjusting either the depth or spatial size of the model.

It was however observed that some uncertainty propagates through the network using these techniques. For instance, assume the following 4x4 matrix:

$$
\begin{bmatrix}
7 & 7 \\
4 & 5
\end{bmatrix}
$$

A max pooling layer needs to choose which of the 7s to keep, and later, during upsampling, remember the index where the 7 was found. Different implementations can cause different issues.

## Approach:

Observations from Dr. Glatard's lab suggest that replacing such occurrences with NaN values could potentially improve computational speed during both training and inference, while preserving the accuracy of the model. The logic is that parts of images with redundant information (e.g., the background) could be disregarded during the convolutional operation, as image patches containing a high count of NaN values might be ignored. Likewise, this will solve the problem mentioned above. This is hypothesized to accelerate the calculations in convolutional operations.

A Python implementation of this NaN pooling has already been developed with success in the lab, which showed promising results. However, to fully harness the potential of this approach, it must be adapted to PyTorch's C/C++ framework.

Hence, the following project is proposed: Can a C/C++ PyTorch implementation of this solution improve the speed of inference and training of CNN models and prevent uncertainty propagation while conserving current accuracy benchmarks?

## Works Cited:

[1] Alzubaidi, L., Zhang, J., Humaidi, A.J. et al. Review of deep learning: concepts, CNN architectures, challenges, applications, future directions. J Big Data 8, 53 (2021). https://doi.org/10.1186/s40537-021-00444-8

[2] Zhao, L., Zhang, Z. A improved pooling method for convolutional neural networks. Sci Rep 14, 1589 (2024). https://doi.org/10.1038/s41598-024-51258-6
