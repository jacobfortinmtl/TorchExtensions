#include <torch/extension.h>
#include <optional>
#include <iostream>
#include <tuple>

// Struct to hold pooling attributes
struct PoolingAttributes {
    int batch_size;          // Number of samples in a batch
    int channels;            // Number of channels in the input tensor
    int input_height;        // Height of the input tensor
    int input_width;         // Width of the input tensor
    int pool_height;         // Height of the pooling window
    int pool_width;          // Width of the pooling window
    int stride_height;       // Height of the stride
    int stride_width;        // Width of the stride
    float threshold;         // Threshold to determine if a max value is ambiguous
    torch::Tensor input_tensor;  // The input tensor
    torch::Tensor output_tensor; // The output tensor for pooled values
    torch::Tensor index_tensor;  // The output tensor for pooled indices
};

// Function to check for NaNs in the pooling window and update the output tensors
std::tuple<torch::Tensor, torch::Tensor> check_for_nans(
    PoolingAttributes& attrs,
    torch::Tensor window, 
    torch::Tensor maxval, 
    torch::Tensor max_index,
    int c, 
    int i, 
    int j
) {
    // Convert 1D indices from the pooling window to 2D indices
    max_index = torch::stack({max_index / attrs.pool_height, max_index % attrs.pool_width}, 1);
    // If any of the max values are NaN, replace NaNs in the window with -infinity and recompute max values
    if (torch::isnan(maxval).any().item<bool>()) {
        window.masked_fill_(torch::isnan(window), -std::numeric_limits<float>::infinity());
        maxval = std::get<0>(torch::max(window.reshape({attrs.batch_size, -1}), 1));
    }
    
    
    // Check if multiple values are close to the max value
    auto reshaped_maxval = maxval.unsqueeze(1).unsqueeze(2);

    // Perform the isclose comparison with the specified rtol and equal_nan=True
    auto close_to_max = torch::isclose(window, reshaped_maxval, 1e-7, 1e-7, true);
    auto check_multi_max = torch::sum(close_to_max, {1, 2});

    // changing check_mulit_max to a ratio
    check_multi_max = check_multi_max.to(torch::kFloat32) / (attrs.pool_height * attrs.pool_width);
    

    // If the proportion of close values exceeds the threshold, set the max value to NaN
    // std::cout << "check_multi_max: " << check_multi_max << std::endl;   
    if ((check_multi_max >= attrs.threshold).any().item<bool>()) {
        auto nan_tensor = torch::full_like(maxval, std::numeric_limits<float>::quiet_NaN());
        maxval = torch::where(check_multi_max > attrs.threshold, nan_tensor, maxval);
    }

    // Type-casting to int
    max_index = max_index.toType(torch::kInt64);

    // Find new index of max value if it has changed and is not NaN
    // std::cout << "max_index: " << max_index << std::endl;
    auto non_zero_elems = torch::where(window == maxval)[0].numel();
    if (non_zero_elems != 0) {
        auto max_vals = torch::max(
            window.masked_fill(torch::isnan(window), -std::numeric_limits<float>::infinity()).view({attrs.batch_size, -1}), 1
        );
        max_index = std::get<1>(max_vals).toType(torch::kInt64);
        // std::cout << "max_index during: " << max_index << std::endl;
        max_index = torch::stack({max_index / attrs.pool_width, max_index % attrs.pool_width}, 1);
    }
    max_index = max_index.toType(torch::kInt64);
    // Convert the 2D indices back to 1D
    auto max_index_1d = 
        (i * attrs.stride_height + max_index.index({torch::indexing::Slice(), 0})) * attrs.input_width + 
        (j * attrs.stride_width + max_index.index({torch::indexing::Slice(), 1}));

    

    // Update the output tensors with the computed max values and indices
    attrs.output_tensor.index_put_({
        torch::indexing::Slice(), 
        c, 
        i, 
        j
        }, maxval.view({attrs.batch_size}));
    attrs.index_tensor.index_put_({
        torch::indexing::Slice(), 
        c, 
        i, 
        j}, max_index_1d.view({attrs.batch_size}));

    return std::make_tuple(attrs.output_tensor, attrs.index_tensor);
}

// Main function to perform NaN-aware max pooling
std::tuple<torch::Tensor, torch::Tensor> NaNPool2d(
    torch::Tensor input_tensor, 
    std::tuple<int, int> pool_size, 
    float threshold = 0.5, 
    std::optional<std::tuple<int, int>> strides = std::nullopt
) {
    // Extract the size of the input tensor
    auto size = input_tensor.sizes();
    PoolingAttributes attrs;
    attrs.batch_size = size[0];
    attrs.channels = size[1];
    attrs.input_height = size[2];
    attrs.input_width = size[3];

    // Unpack the pool size
    std::tie(attrs.pool_height, attrs.pool_width) = pool_size;

    // Unpack or set default strides
    if (strides) {
        std::tie(attrs.stride_height, attrs.stride_width) = strides.value();
    } else {
        std::tie(attrs.stride_height, attrs.stride_width) = pool_size;
    }

    // Set the threshold and input tensor
    attrs.threshold = threshold;
    attrs.input_tensor = input_tensor;

    // Calculate the output dimensions
    int output_height = (attrs.input_height - attrs.pool_height) / attrs.stride_height + 1;
    int output_width = (attrs.input_width - attrs.pool_width) / attrs.stride_width + 1;

    // Initialize the output tensors for pooled values and indices
    attrs.output_tensor = torch::zeros({attrs.batch_size, attrs.channels, output_height, output_width});
    attrs.index_tensor = torch::zeros({attrs.batch_size, attrs.channels, output_height, output_width}, torch::kInt64);

    // Perform max pooling with NaN handling
    for (int c = 0; c < attrs.channels; c++) {
        for (int i = 0; i < output_height; i++) {
            for (int j = 0; j < output_width; j++) {
                // Extract the current pooling window
                torch::Tensor window = attrs.input_tensor.index({
                    torch::indexing::Slice(),
                    c,
                    torch::indexing::Slice(i * attrs.stride_height, i * attrs.stride_height + attrs.pool_height),
                    torch::indexing::Slice(j * attrs.stride_width, j * attrs.stride_width + attrs.pool_width)
                });

                // Reshape the window for max pooling
                torch::Tensor reshaped_window = window.reshape({attrs.batch_size, -1});

                // Compute the max values and their indices in the window
                auto max_result = torch::max(reshaped_window, 1);
                auto max_values = std::get<0>(max_result);
                auto max_indices = std::get<1>(max_result);

                // Check for NaNs and update the output tensors
                std::tie(attrs.output_tensor, attrs.index_tensor) = check_for_nans(
                    attrs,
                    window, 
                    max_values, 
                    max_indices,
                    c, 
                    i, 
                    j
                );
            }
        }
    }

    return std::make_tuple(attrs.output_tensor, attrs.index_tensor);
}

// Bind the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("NaNPool2d", &NaNPool2d, "Create a NaN pooling layer. Returns a tuple of tensors.");
}