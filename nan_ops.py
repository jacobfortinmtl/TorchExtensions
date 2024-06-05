import random
import numpy as np
import torch
import torch.nn.init as init
from torch import nn
from typing import Tuple
from math import prod  
from torch.nn import functional as F

class NaNConv2d(nn.Module):
    def __init__(
        self,
        train: bool = False,
        bias_presence: bool = True,
        padding: int = 0,
        stride: int = 1,
        threshold: float = 0.5,
        in_channels: int = None,
        out_channels: int = None,
        kernel_size: Tuple = None,
        kernel: torch.Tensor = None,
        bias: torch.Tensor = None,
    ):
        """
        Initializes the NaNConv2d layer.

        Args:
            train (bool): Whether to train the model.
            bias_presence (bool): Whether bias is present in the layer.
            padding (int): Padding size.
            stride (int): Stride size.
            threshold (float): Threshold for NaN ratio.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple): Size of the convolutional kernel.
            kernel (torch.Tensor): Initial kernel weights.
            bias (torch.Tensor): Initial bias values.
        """

        super().__init__()
        # kernel = torch.flipud(torch.fliplr(kernel))
        self.stride = int(stride)
        self.padding = int(padding)
        self.threshold = threshold

        if train is False:
            self.inference(kernel, bias)
        else:
            self.trainer(in_channels, out_channels, kernel_size, bias_presence)

        # print(f'Kernel {self.kernel.shape} Stride {self.stride}, padding {self.padding}, bias {self.bias.shape}, in_channels {self.image_padded.shape}, out_channels {self.output.shape}')

    def inference(self, kernel: torch.Tensor, bias: torch.Tensor) -> None:
        """
        Initializes the NaNConv2d layer at inference with the provided kernel and bias from the original model.

        Args:
            kernel (torch.Tensor): 4D tensor representing the convolutional kernel.
            bias (torch.Tensor): 1D tensor representing the bias values.

        Returns:
            None
        """
        self.bias = bias
        self.kernel = kernel
        self.out_channels, _, self.kernel_height, self.kernel_width = kernel.shape
        # Force values to int
        self.out_channels = int(self.out_channels)
        self.kernel_height = int(self.kernel_height)
        self.kernel_width = int(self.kernel_width)

    # UNDER WORK
    def trainer(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], bias_presence: bool) -> None:
        """
        Initialize the NaNConv2d layer for training a model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple[int, int]): Size of the convolutional kernel (height, width).
            bias_presence (bool): Whether bias term should be included.

        Returns:
            None
        """

        self.out_channels = int(out_channels)
        self.kernel_height, self.kernel_width = kernel_size
        # Force values to int
        self.kernel_height = int(self.kernel_height)
        self.kernel_width = int(self.kernel_width)

        # Initialize kernel parameters
        self.kernel = nn.Parameter(
            init.xavier_normal_(torch.zeros((self.out_channels, in_channels, self.kernel_height, self.kernel_width)))
        )
        # Initialize bias parameters if bias is present
        if bias_presence:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.bias = None

    # NO LONGER IN USE
    def choose_probability_large(
        self, input: torch.Tensor, kernel: torch.Tensor, min_val: float, max_val: float
    ) -> torch.Tensor:
        """
        Choose probabilities for replacing NaN values with random values for window across channels and batch.

        Args:
            input (torch.Tensor): Input tensor.
            kernel (torch.Tensor): Kernel tensor.
            min_val (float): Minimum value.
            max_val (float): Maximum value.

        Returns:
            torch.Tensor: Processed tensor with NaN values replaced.
        """
        # If all NaN, return input as is
        if torch.isnan(input).all():
            return input
        elif not torch.isnan(input).any():
            return input

        # Flatten input and kernel
        input_flat = input.ravel()  # view(-1)
        kernel_flat = kernel.ravel()  # view(-1)

        # Calculate probabilities for NaN values
        distance_to_max = max_val - kernel_flat
        total_distance = max_val - min_val
        probabilities = 1 - distance_to_max / total_distance

        # Create a mask for NaN values
        nan_mask = torch.isnan(input_flat)

        # Indices of NaN values
        nan_indices = torch.nonzero(nan_mask).squeeze()

        # Generate random values from the distribution
        hist, bin_edges = torch.histogram(input_flat[~nan_mask], bins=5, density=True)
        random_values = torch.tensor(
            random.choices(bin_edges[:-1], weights=(hist / torch.sum(hist)), k=nan_indices.numel())
        )

        # Replace NaN values with random values
        # input_flat[nan_indices] = random_values

        probabilities = torch.mean(probabilities.view(kernel.shape), axis=0).ravel()

        result = torch.where(
            torch.isnan(input_flat) & (probabilities > random.random()),
            random_values[torch.randint(len(random_values), size=input_flat.shape)],
            input_flat,
        )

        return result.view(input.shape)

    def get_window(self, i: int, j: int) -> torch.Tensor:
        """
        Get the receptive field window at position (i, j).

        Args:
            i (int): Horizontal index.
            j (int): Vertical index.

        Returns:
            torch.Tensor: Receptive field window.
        """
        x_start = i * self.stride
        y_start = j * self.stride

        # Calculate the ending point of the receptive field
        x_end = min(x_start + self.kernel_height, self.in_height + 2 * self.padding)
        y_end = min(y_start + self.kernel_width, self.in_width + 2 * self.padding)

        # Extract the image patch
        return self.image_padded[:, :, x_start:x_end, y_start:y_end]

    def apply_threshold(self, kernel: torch.Tensor, i: int, j: int) -> None:
        """
        Apply NaN thresholding to the output tensor window at position (i, j).

        Args:
            kernel (torch.Tensor): Convolution kernel.
            i (int): Horizontal index.
            j (int): Vertical index.
        """

        image_patch = self.get_window(i, j)
        # image_patch = choose_probability_large(image_patch, kernel, torch.min(kernel), torch.max(kernel))

        # Calculate NaN ratio
        try:
            nan_ratio = torch.sum(torch.isnan(image_patch)).item() / image_patch.numel()
        except ZeroDivisionError:
            nan_ratio = 1

        # Use NaN ratio to determine whether NaNs are ignored or convolution output
        if nan_ratio >= self.threshold:
            self.output[:, :, i, j] = float("nan")
        else:
            # self.output[:, :, i, j] = torch.nansum(image_patch.unsqueeze(1) * kernel.unsqueeze(0), dim=(2, 3, 4))
            self.output[:, :, i, j] = F.conv2d(torch.nan_to_num(image_patch), kernel, stride=1, padding=0).squeeze(dim=-1).squeeze(dim=-1)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NaNConv module.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output tensor window after applying NaNConv operation.
        """

        # Pad the input image
        self.image_padded = torch.nn.functional.pad(image, (self.padding, self.padding, self.padding, self.padding))

        # Get dimensions
        self.batch_size, _, self.in_height, self.in_width = self.image_padded.shape

        self.x_img, self.y_img = image.shape[-2:]

        self.out_height = int(((self.x_img + 2 * self.padding - 1 * (self.kernel_height - 1)) - 1) / self.stride) + 1
        self.out_width = int(((self.y_img + 2 * self.padding - 1 * (self.kernel_width - 1)) - 1) / self.stride) + 1

        # Initialize output tensor
        self.output = torch.zeros(self.batch_size, self.out_channels, self.out_height, self.out_width)

        _ = [self.apply_threshold(self.kernel, i, j) for i in range(self.out_height) for j in range(self.out_width)]
        # self.output = torch.Tensor([self.apply_threshold(self.kernel, i, j) for i in range(self.out_height) for j in range(self.out_width)])

        if self.bias is not None:
            self.output += self.bias.view(1, -1, 1, 1)

        return self.output

class NormalConv2d(nn.Module):
    def __init__(
        self,
        train: bool = False,
        bias_presence: bool = True,
        padding: int = 0,
        stride: int = 1,
        threshold: float = 0.5,
        in_channels: int = None,
        out_channels: int = None,
        kernel_size: Tuple = None,
        kernel: torch.Tensor = None,
        bias: torch.Tensor = None,
    ):
        """
        Initializes the NaNConv2d layer.

        Args:
            train (bool): Whether to train the model.
            bias_presence (bool): Whether bias is present in the layer.
            padding (int): Padding size.
            stride (int): Stride size.
            threshold (float): Threshold for NaN ratio.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple): Size of the convolutional kernel.
            kernel (torch.Tensor): Initial kernel weights.
            bias (torch.Tensor): Initial bias values.
        """

        super().__init__()
        # kernel = torch.flipud(torch.fliplr(kernel))
        self.stride = int(stride)
        self.padding = int(padding)
        self.threshold = threshold

        if train is False:
            self.inference(kernel, bias)
        else:
            self.trainer(in_channels, out_channels, kernel_size, bias_presence)

        # print(f'Kernel {self.kernel.shape} Stride {self.stride}, padding {self.padding}, bias {self.bias.shape}, in_channels {self.image_padded.shape}, out_channels {self.output.shape}')

    def inference(self, kernel: torch.Tensor, bias: torch.Tensor) -> None:
        """
        Initializes the NaNConv2d layer at inference with the provided kernel and bias from the original model.

        Args:
            kernel (torch.Tensor): 4D tensor representing the convolutional kernel.
            bias (torch.Tensor): 1D tensor representing the bias values.

        Returns:
            None
        """
        self.bias = bias
        self.kernel = kernel
        self.out_channels, _, self.kernel_height, self.kernel_width = kernel.shape
        # Force values to int
        self.out_channels = int(self.out_channels)
        self.kernel_height = int(self.kernel_height)
        self.kernel_width = int(self.kernel_width)

    # UNDER WORK
    def trainer(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], bias_presence: bool) -> None:
        """
        Initialize the NaNConv2d layer for training a model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple[int, int]): Size of the convolutional kernel (height, width).
            bias_presence (bool): Whether bias term should be included.

        Returns:
            None
        """

        self.out_channels = int(out_channels)
        self.kernel_height, self.kernel_width = kernel_size
        # Force values to int
        self.kernel_height = int(self.kernel_height)
        self.kernel_width = int(self.kernel_width)

        # Initialize kernel parameters
        self.kernel = nn.Parameter(
            init.xavier_normal_(torch.zeros((self.out_channels, in_channels, self.kernel_height, self.kernel_width)))
        )
        # Initialize bias parameters if bias is present
        if bias_presence:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.bias = None

    def get_window(self, i: int, j: int) -> torch.Tensor:
        """
        Get the receptive field window at position (i, j).

        Args:
            i (int): Horizontal index.
            j (int): Vertical index.

        Returns:
            torch.Tensor: Receptive field window.
        """
        x_start = i * self.stride
        y_start = j * self.stride

        # Calculate the ending point of the receptive field
        x_end = min(x_start + self.kernel_height, self.in_height + 2 * self.padding)
        y_end = min(y_start + self.kernel_width, self.in_width + 2 * self.padding)

        # Extract the image patch
        return self.image_padded[:, :, x_start:x_end, y_start:y_end]

    def apply_threshold(self, kernel: torch.Tensor, i: int, j: int) -> None:
        """
        Apply NaN thresholding to the output tensor window at position (i, j).

        Args:
            kernel (torch.Tensor): Convolution kernel.
            i (int): Horizontal index.
            j (int): Vertical index.
        """

        image_patch = self.get_window(i, j)
        # image_patch = choose_probability_large(image_patch, kernel, torch.min(kernel), torch.max(kernel))

        # self.output[:, :, i, j] = torch.sum(image_patch.unsqueeze(1) * kernel.unsqueeze(0), dim=(2, 3, 4))
        self.output[:, :, i, j] = F.conv2d(image_patch, kernel, stride=1, padding=0).squeeze(dim=-1).squeeze(dim=-1)


    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NaNConv module.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output tensor window after applying NaNConv operation.
        """

        # Pad the input image
        self.image_padded = torch.nn.functional.pad(image, (self.padding, self.padding, self.padding, self.padding))

        # Get dimensions
        self.batch_size, _, self.in_height, self.in_width = self.image_padded.shape

        self.x_img, self.y_img = image.shape[-2:]

        self.out_height = int(((self.x_img + 2 * self.padding - 1 * (self.kernel_height - 1)) - 1) / self.stride) + 1
        self.out_width = int(((self.y_img + 2 * self.padding - 1 * (self.kernel_width - 1)) - 1) / self.stride) + 1

        # Initialize output tensor
        self.output = torch.zeros(self.batch_size, self.out_channels, self.out_height, self.out_width)

        _ = [self.apply_threshold(self.kernel, i, j) for i in range(self.out_height) for j in range(self.out_width)]
        # self.output = torch.Tensor([self.apply_threshold(self.kernel, i, j) for i in range(self.out_height) for j in range(self.out_width)])

        if self.bias is not None:
            self.output += self.bias.view(1, -1, 1, 1)

        return self.output


class NoArgMaxIndices(BaseException):
    def __init__(self):
        super(NoArgMaxIndices, self).__init__(
            "no argmax indices: batch_argmax requires non-batch shape to be non-empty")


class NaNPool2d:

    def __init__(self, max_threshold: float = 1):
        """
        Initializes the NaNPool2d object.

        Args:
            max_threshold (float, optional): Max threshold value for determining multiple max value occurrence ratio. Defaults to 0.5.
        """
        self.max_threshold = max_threshold

    # CURRENTLY UNUSED
    ## unravel_index and batch_argmax are from https://stackoverflow.com/questions/39458193/using-list-tuple-etc-from-typing-vs-directly-referring-type-as-list-tuple-etc
    ## they generalize well to multi-dimensions in theory but are they necessary? Window is always 3D
    def unravel_index(self, 
        indices: torch.LongTensor,
        shape: Tuple[int, ...],
    ) -> torch.LongTensor:
        r"""Converts flat indices into unraveled coordinates in a target shape.

        This is a `torch` implementation of `numpy.unravel_index`.

        Args:
            indices: A tensor of (flat) indices, (*, N).
            shape: The targeted shape, (D,).

        Returns:
            The unraveled coordinates, (*, N, D).
        """

        coord = []

        for dim in reversed(shape):
            coord.append(indices % dim)
            indices = indices // dim

        coord = torch.stack(coord[::-1], dim=-1)

        return coord

    # CURRENTLY UNUSED
    def batch_argmax(self, tensor, batch_dim=1):
        """
        Assumes that dimensions of tensor up to batch_dim are "batch dimensions"
        and returns the indices of the max element of each "batch row".
        More precisely, returns tensor `a` such that, for each index v of tensor.shape[:batch_dim], a[v] is
        the indices of the max element of tensor[v].
        """
        if batch_dim >= len(tensor.shape):
            raise NoArgMaxIndices()
        batch_shape = tensor.shape[:batch_dim]
        non_batch_shape = tensor.shape[batch_dim:]
        flat_non_batch_size = prod(non_batch_shape)
        tensor_with_flat_non_batch_portion = tensor.reshape(*batch_shape, flat_non_batch_size)

        dimension_of_indices = len(non_batch_shape)

        # We now have each batch row flattened in the last dimension of tensor_with_flat_non_batch_portion,
        # so we can invoke its argmax(dim=-1) method. However, that method throws an exception if the tensor
        # is empty. We cover that case first.
        if tensor_with_flat_non_batch_portion.numel() == 0:
            # If empty, either the batch dimensions or the non-batch dimensions are empty
            batch_size = prod(batch_shape)
            if batch_size == 0:  # if batch dimensions are empty
                # return empty tensor of appropriate shape
                batch_of_unraveled_indices = torch.ones(*batch_shape, dimension_of_indices).long()  # 'ones' is irrelevant as it will be empty
            else:  # non-batch dimensions are empty, so argmax indices are undefined
                raise NoArgMaxIndices()
        else:   # We actually have elements to maximize, so we search for them
            indices_of_non_batch_portion = tensor_with_flat_non_batch_portion.argmax(dim=-1)
            batch_of_unraveled_indices = self.unravel_index(indices_of_non_batch_portion, non_batch_shape)

        if dimension_of_indices == 1:
            # above function makes each unraveled index of a n-D tensor a n-long tensor
            # however indices of 1D tensors are typically represented by scalars, so we squeeze them in this case.
            batch_of_unraveled_indices = batch_of_unraveled_indices.squeeze(dim=-1)
        return batch_of_unraveled_indices

    def check_for_nans(self, c, i, j, window, maxval, max_index):

        # Converting 1d indices from pool window to 2d indices
        max_index = torch.stack((max_index // self.pool_height, max_index % self.pool_width), dim=1)

        if torch.isnan(maxval).any():
            window = window.masked_fill(torch.isnan(window), float("-inf"))
            maxval = torch.max(window.reshape(self.batch_size, -1), dim=1)[0] 


        # Strict approach to identifying multiple max values
        # check_multi_max = torch.sum(window == maxval[:, None, None], axis=(1, 2))
        # Less restrictive more theoretically stable approach
        check_multi_max = torch.sum(
            torch.isclose(window, maxval[:, None, None], rtol=1e-7, equal_nan=True), axis=(1, 2)
        )

        # Reduce multiple max value counts to ratios in order to use passed max threshold value
        # check_multi_max = check_multi_max / (window.shape[-1] * window.shape[-2])

        if (check_multi_max > self.max_threshold).any():
            maxval = torch.where(check_multi_max > self.max_threshold, np.nan, maxval)

        # Find new index of max value if it has changed and is not NaN
        if torch.where(window == maxval)[0].numel() != 0: 
            max_index = torch.max(window.masked_fill(torch.isnan(window), float('-inf')).reshape(self.batch_size, -1), dim=1)[1]
            max_index = torch.stack((max_index // self.pool_height, max_index % self.pool_width), dim=1)

        # Calculate the indices for 1D representation
        max_index_1d = (i * self.stride_height + (max_index[:, 0])) * self.input_width + (
            j * self.stride_width + (max_index[:, 1])
        )


        self.output_array[:, c, i, j] = maxval
        self.index_array[:, c, i, j] = max_index_1d


    def __call__(self, input_array: torch.Tensor, pool_size: Tuple, strides: Tuple = None) -> Tuple:
        """
        Perform NaN-aware max pooling on the input array.

        Args:
            input_array (torch.Tensor): Input tensor of shape (batch_size, channels, input_height, input_width).
            pool_size (tuple): Size of the pooling window (pool_height, pool_width).
            strides (tuple, optional): Strides for pooling (stride_height, stride_width). Defaults to None.

        Returns:
            tuple: A tuple containing output array and index array after pooling.
        """

        batch_size, channels, input_height, input_width = input_array.shape
        # Force values to int
        self.batch_size = int(batch_size)
        channels = int(channels)
        self.input_height = int(input_height)
        self.input_width = int(input_width)

        pool_height, pool_width = pool_size
        # Force values to int
        self.pool_height = int(pool_height)
        self.pool_width = int(pool_width)

        if strides:
            stride_height, stride_width = strides

        else:
            stride_height, stride_width = pool_size

        # Force values to int
        self.stride_height = int(stride_height)
        self.stride_width = int(stride_width)

        # Calculate simplified intensity distribution of the layer
        self.min_intensity = torch.min(input_array)
        self.max_intensity = torch.max(input_array)

        # Calculate the output dimensions
        output_height = int((input_height - pool_height) // stride_height + 1)
        output_width = int((input_width - pool_width) // stride_width + 1)

        # Initialize output arrays for pooled values and indices
        self.output_array = torch.zeros((self.batch_size, channels, output_height, output_width))
        self.index_array = torch.zeros((self.batch_size, channels, output_height, output_width), dtype=torch.int64)


        # Perform max pooling with list comprehensions
        for c in range(channels):

            # Create a list of tuples with pooled values and indices
            values_and_indices = [
                self.check_for_nans(c, i, j, window, torch.max(window.reshape(self.batch_size, -1), dim=1)[0], torch.max(window.reshape(self.batch_size, -1), dim=1)[1])
                for i in range(output_height)
                for j in range(output_width)
                for window in [
                    input_array[
                        :,
                        c,
                        i * stride_height : i * stride_height + pool_height,
                        j * stride_width : j * stride_width + pool_width,
                    ]
                ]
            ]

        return torch.Tensor(self.output_array), torch.Tensor(self.index_array).type(torch.int64)


class NormalPool2d:

    def __init__(self, max_threshold: float = 1):
        """
        Initializes the NaNPool2d object.

        Args:
            max_threshold (float, optional): Max threshold value for determining multiple max value occurrence ratio. Defaults to 0.5.
        """
        self.max_threshold = max_threshold

    def __call__(self, input_array: torch.Tensor, pool_size: Tuple, strides: Tuple = None) -> Tuple:
        """
        Perform NaN-aware max pooling on the input array.

        Args:
            input_array (torch.Tensor): Input tensor of shape (batch_size, channels, input_height, input_width).
            pool_size (tuple): Size of the pooling window (pool_height, pool_width).
            strides (tuple, optional): Strides for pooling (stride_height, stride_width). Defaults to None.

        Returns:
            tuple: A tuple containing output array and index array after pooling.
        """

        batch_size, channels, input_height, input_width = input_array.shape
        # Force values to int
        batch_size = int(batch_size)
        channels = int(channels)
        input_height = int(input_height)
        input_width = int(input_width)

        pool_height, pool_width = pool_size
        # Force values to int
        pool_height = int(pool_height)
        pool_width = int(pool_width)

        if strides:
            stride_height, stride_width = strides

        else:
            stride_height, stride_width = pool_size

        # Force values to int
        stride_height = int(stride_height)
        stride_width = int(stride_width)

        # Calculate simplified intensity distribution of the layer
        self.min_intensity = torch.min(input_array)
        self.max_intensity = torch.max(input_array)

        # Calculate the output dimensions
        output_height = int((input_height - pool_height) // stride_height + 1)
        output_width = int((input_width - pool_width) // stride_width + 1)

        # Initialize output arrays for pooled values and indices
        output_array = torch.zeros((batch_size, channels, output_height, output_width))
        index_array = torch.zeros((batch_size, channels, output_height, output_width), dtype=torch.int64)

        # Perform max pooling with list comprehensions
        for c in range(channels):

            # Create a list of tuples with pooled values and indices
            values_and_indices = [
                (torch.max(window.reshape(batch_size, -1), dim=1)[0], torch.argmax(window, dim=1))
                for i in range(output_height)
                for j in range(output_width)
                for window in [
                    input_array[
                        :,
                        c,
                        i * stride_height : i * stride_height + pool_height,
                        j * stride_width : j * stride_width + pool_width,
                    ]
                ]
            ]

            # Handle NaNs and probabilities for choosing max value
            for k, (maxval, max_index) in enumerate(values_and_indices):
                # Re-initialize the window within the second for-loop
                i = k // output_width
                j = k % output_width
                window = input_array[
                    :,
                    c,
                    i * stride_height : i * stride_height + pool_height,
                    j * stride_width : j * stride_width + pool_width,
                ]

                # if torch.isnan(maxval).any():
                #     window = window.masked_fill(torch.isnan(window), float("-inf"))
                #     maxval = torch.max(window).reshape(1)

                # # Strict approach to identifying multiple max values
                # # check_multi_max = torch.sum(window == maxval[:, None, None], axis=(1, 2))
                # # Less restrictive more theoretically stable approach
                # check_multi_max = torch.sum(
                #     torch.isclose(window, maxval[:, None, None], rtol=1e-7, equal_nan=True), axis=(1, 2)
                # )

                # # Reduce multiple max value counts to ratios in order to use passed max threshold value
                # # check_multi_max = check_multi_max / (window.shape[-1] * window.shape[-2])

                # if (check_multi_max > self.max_threshold).any():
                #     maxval = torch.where(check_multi_max > self.max_threshold, np.nan, maxval)

                # Calculate the indices for 1D representation
                max_index_1d = (i * stride_height + (max_index[:, 0] // pool_width)) * input_width + (
                    j * stride_width + (max_index[:, 1] % pool_width)
                )

                # Update output arrays
                output_array[:, c, i, j] = maxval
                index_array[:, c, i, j] = max_index_1d
            # break

        return torch.Tensor(output_array), torch.Tensor(index_array).type(torch.int64)



# Conv2d Skipped Operation Counter -- not to be implemented in CPP
def count_skip_conv2d(image, kernel, padding=0, stride=1, threshold=0.5):


    # kernel = torch.flipud(torch.fliplr(kernel))

    # Pad the input image
    image_padded = torch.nn.functional.pad(image, (padding, padding, padding, padding))

    # Get dimensions
    batch_size, in_channels, in_height, in_width = image_padded.shape
    out_channels, _, kernel_height, kernel_width = kernel.shape

    x_img, y_img = image.shape[-2:]

    out_height = int((( x_img + 2*padding - 1*(kernel_height - 1) ) -1)/stride ) + 1
    out_width = int((( y_img + 2*padding - 1*(kernel_width - 1) ) -1)/stride ) + 1

    print(out_height, out_width)
    skip = 0
    total=0

    # Perform convolution
    for c in range(in_channels):
        for i in range(0, out_height):

            for j in range(0, out_width):

                #PUT IN FUNCTION FOR LIST COMPREHENSION
                x_start = i * stride
                y_start = j * stride
                # Calculate the ending point of the receptive field
                x_end = min(x_start + kernel_height, in_height + 2 * padding)
                y_end = min(y_start + kernel_width, in_width + 2 * padding)

                # Extract the image patch
                image_patch = image_padded[:, c, x_start:x_end, y_start:y_end]
                
                try:
                    # print(torch.sum(torch.isnan(image_patch)).item(), torch.sum(~torch.isnan(image_patch)).item())
                    # nan_ratio = torch.sum(torch.isnan(image_patch)).item() / torch.sum(~torch.isnan(image_patch)).item() 
                    nan_ratio = torch.sum(torch.isnan(image_patch)).item() / image_patch.numel() 
                except ZeroDivisionError:
                    nan_ratio = 1
                
                if nan_ratio >= threshold: 
                    skip += 1
                total+=1
                
        
    return skip, total

class NaNConv2dV2(nn.Module):
    def __init__(
        self,
        train: bool = False,
        bias_presence: bool = True,
        padding: int = 0,
        stride: int = 1,
        threshold: float = 0.5,
        in_channels: int = None,
        out_channels: int = None,
        kernel_size: Tuple = None,
        kernel: torch.Tensor = None,
        bias: torch.Tensor = None,
    ):
        """
        Initializes the NaNConv2d layer.

        Args:
            train (bool): Whether to train the model.
            bias_presence (bool): Whether bias is present in the layer.
            padding (int): Padding size.
            stride (int): Stride size.
            threshold (float): Threshold for NaN ratio.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple): Size of the convolutional kernel.
            kernel (torch.Tensor): Initial kernel weights.
            bias (torch.Tensor): Initial bias values.
        """

        super().__init__()
        # kernel = torch.flipud(torch.fliplr(kernel))
        self.stride = int(stride)
        self.padding = int(padding)
        self.threshold = threshold

        if train is False:
            self.inference(kernel, bias)
        else:
            self.trainer(in_channels, out_channels, kernel_size, bias_presence)

        # print(f'Kernel {self.kernel.shape} Stride {self.stride}, padding {self.padding}, bias {self.bias.shape}, in_channels {self.image_padded.shape}, out_channels {self.output.shape}')

    def inference(self, kernel: torch.Tensor, bias: torch.Tensor) -> None:
        """
        Initializes the NaNConv2d layer at inference with the provided kernel and bias from the original model.

        Args:
            kernel (torch.Tensor): 4D tensor representing the convolutional kernel.
            bias (torch.Tensor): 1D tensor representing the bias values.

        Returns:
            None
        """
        self.bias = bias
        self.kernel = kernel
        self.out_channels, _, self.kernel_height, self.kernel_width = kernel.shape
        # Force values to int
        self.out_channels = int(self.out_channels)
        self.kernel_height = int(self.kernel_height)
        self.kernel_width = int(self.kernel_width)

    # UNDER WORK
    def trainer(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], bias_presence: bool) -> None:
        """
        Initialize the NaNConv2d layer for training a model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple[int, int]): Size of the convolutional kernel (height, width).
            bias_presence (bool): Whether bias term should be included.

        Returns:
            None
        """

        self.out_channels = int(out_channels)
        self.kernel_height, self.kernel_width = kernel_size
        # Force values to int
        self.kernel_height = int(self.kernel_height)
        self.kernel_width = int(self.kernel_width)

        # Initialize kernel parameters
        self.kernel = nn.Parameter(
            init.xavier_normal_(torch.zeros((self.out_channels, in_channels, self.kernel_height, self.kernel_width)))
        )
        # Initialize bias parameters if bias is present
        if bias_presence:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.bias = None

    # NO LONGER IN USE
    def choose_probability_large(
        self, input: torch.Tensor, kernel: torch.Tensor, min_val: float, max_val: float
    ) -> torch.Tensor:
        """
        Choose probabilities for replacing NaN values with random values for window across channels and batch.

        Args:
            input (torch.Tensor): Input tensor.
            kernel (torch.Tensor): Kernel tensor.
            min_val (float): Minimum value.
            max_val (float): Maximum value.

        Returns:
            torch.Tensor: Processed tensor with NaN values replaced.
        """
        # If all NaN, return input as is
        if torch.isnan(input).all():
            return input
        elif not torch.isnan(input).any():
            return input

        # Flatten input and kernel
        input_flat = input.ravel()  # view(-1)
        kernel_flat = kernel.ravel()  # view(-1)

        # Calculate probabilities for NaN values
        distance_to_max = max_val - kernel_flat
        total_distance = max_val - min_val
        probabilities = 1 - distance_to_max / total_distance

        # Create a mask for NaN values
        nan_mask = torch.isnan(input_flat)

        # Indices of NaN values
        nan_indices = torch.nonzero(nan_mask).squeeze()

        # Generate random values from the distribution
        hist, bin_edges = torch.histogram(input_flat[~nan_mask], bins=5, density=True)
        random_values = torch.tensor(
            random.choices(bin_edges[:-1], weights=(hist / torch.sum(hist)), k=nan_indices.numel())
        )

        # Replace NaN values with random values
        # input_flat[nan_indices] = random_values

        probabilities = torch.mean(probabilities.view(kernel.shape), axis=0).ravel()

        result = torch.where(
            torch.isnan(input_flat) & (probabilities > random.random()),
            random_values[torch.randint(len(random_values), size=input_flat.shape)],
            input_flat,
        )

        return result.view(input.shape)

    def get_window(self, i: int, j: int) -> torch.Tensor:
        """
        Get the receptive field window at position (i, j).

        Args:
            i (int): Horizontal index.
            j (int): Vertical index.

        Returns:
            torch.Tensor: Receptive field window.
        """
        x_start = i * self.stride
        y_start = j * self.stride

        # Calculate the ending point of the receptive field
        x_end = min(x_start + self.kernel_height, self.in_height + 2 * self.padding)
        y_end = min(y_start + self.kernel_width, self.in_width + 2 * self.padding)

        # Extract the image patch
        return self.image_padded[:, :, x_start:x_end, y_start:y_end]

    def apply_threshold(self, kernel: torch.Tensor, i: int, j: int) -> None:
        """
        Apply NaN thresholding to the output tensor window at position (i, j).

        Args:
            kernel (torch.Tensor): Convolution kernel.
            i (int): Horizontal index.
            j (int): Vertical index.
        """

        image_patch = self.get_window(i, j)
        # image_patch = choose_probability_large(image_patch, kernel, torch.min(kernel), torch.max(kernel))

        nan_mask = torch.isnan(image_patch)

        # Replace NaNs with zeros
        image_patch_nonan = torch.nan_to_num(image_patch)

        # Calculate the ratio of NaNs
        nan_ratio = torch.sum(nan_mask).item() / image_patch.numel()

        # Determine the output based on the NaN ratio
        if nan_ratio >= self.threshold:
            self.output[:, :, i, j] = float("nan")
        else:
            # Perform the convolution operation
            self.output[:, :, i, j] = F.conv2d(image_patch_nonan.unsqueeze(0), kernel.unsqueeze(0), stride=1, padding=0).squeeze()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NaNConv module.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output tensor window after applying NaNConv operation.
        """

        # Pad the input image
        self.image_padded = torch.nn.functional.pad(image, (self.padding, self.padding, self.padding, self.padding))

        # Get dimensions
        self.batch_size, _, self.in_height, self.in_width = self.image_padded.shape

        self.x_img, self.y_img = image.shape[-2:]

        self.out_height = int(((self.x_img + 2 * self.padding - 1 * (self.kernel_height - 1)) - 1) / self.stride) + 1
        self.out_width = int(((self.y_img + 2 * self.padding - 1 * (self.kernel_width - 1)) - 1) / self.stride) + 1

        # Initialize output tensor
        self.output = torch.zeros(self.batch_size, self.out_channels, self.out_height, self.out_width)

        _ = [self.apply_threshold(self.kernel, i, j) for i in range(self.out_height) for j in range(self.out_width)]
        # self.output = torch.Tensor([self.apply_threshold(self.kernel, i, j) for i in range(self.out_height) for j in range(self.out_width)])

        if self.bias is not None:
            self.output += self.bias.view(1, -1, 1, 1)

        return self.output
