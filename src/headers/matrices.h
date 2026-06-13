#ifndef MATRICES_H
#define MATRICES_H

#include "tensor.h"

namespace matrices {

    // Matrix multiplication for N-dimensional tensors.
    // The last two dims are treated as matrix dims: [..., M, K] x [..., K, N] -> [..., M, N]
    // Leading (batch) dimensions must match.
    Tensor matmul(const Tensor& a, const Tensor& b);

    // Element-wise operations (shapes must match)
    Tensor add(const Tensor& a, const Tensor& b);
    Tensor subtract(const Tensor& a, const Tensor& b);
    Tensor hadamard(const Tensor& a, const Tensor& b);  // element-wise multiply

    // Scalar operations
    Tensor scalar_multiply(const Tensor& a, float scalar);
    Tensor scalar_add(const Tensor& a, float scalar);

    // Swaps the last two dimensions: [..., M, N] -> [..., N, M]
    Tensor transpose(const Tensor& a);

    // Applies a function to every element (useful for activations like ReLU/sigmoid)
    Tensor apply(const Tensor& a, float (*func)(float));

    // Sum of all elements (useful for loss reduction)
    float sum(const Tensor& a);

} // namespace matrices

#endif
