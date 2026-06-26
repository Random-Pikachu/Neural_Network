#include "../headers/tensor.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// constructor
Tensor::Tensor(std::vector<int> _shape) : _shape(_shape){
    int total_size = 1;
    for (auto dim: _shape){
        total_size *= dim;
    }

    _data.resize(total_size, 0.0f);
    _strides.resize(_shape.size(), 1);

    int curr_stride = 1;
    for (int i = _shape.size()-1; i>=0; i--){
        _strides[i] = curr_stride;
        curr_stride*=_shape[i];
    }
}


// --- Accessors ---

// Returns the total number of elements in the flat array
int Tensor::size() const {
    return _data.size();
}

// Returns a read-only reference to the shape vector
const std::vector<int>& Tensor::shape() const {
    return _shape;
}

// Returns a read-only reference to the underlying flat data array
const std::vector<float>& Tensor::data() const {
    return _data;
}

// Returns a modifiable reference to the underlying flat data array
std::vector<float>& Tensor::data() {
    return _data;
}

void Tensor::load_data(const std::vector<float> &input_data) {
    if (input_data.size() != _data.size()){
        throw std::runtime_error("Data size doesnt match with tensors total size");
    }
    _data = input_data;
}

int Tensor::get_flat_index(const std::vector<int> &coordinates) const {
    if (coordinates.size() != _shape.size()){
        throw std::runtime_error("Coordinates shape doesn't match with the tensor shape");
    }

    int flatIdx = 0;
    for (int i = 0; i<coordinates.size(); i++){
        flatIdx+=(coordinates[i] * _strides[i]);
    }

    return flatIdx;
}


std::vector<int> Tensor::get_coordinates(int flat_index) const{
    if (flat_index >= _data.size() || flat_index<0) {
        throw std::runtime_error("Index out of range!");
    }
    std::vector<int> coord(_shape.size());
    int tmp = flat_index;
    for (int i = _shape.size()-1; i>=0; i--){
        coord[i] = tmp%_shape[i];
        tmp/=_shape[i];
    }

    return coord;
}

float Tensor::at(const std::vector<int> &coordinates) const {
    return _data[get_flat_index(coordinates)];
}

void Tensor::set(const std::vector<int> &coordinates, float value) {
    _data[get_flat_index(coordinates)] = value;
}

std::vector<int> Tensor::get_batch_shape() const {
    if (_shape.size() < 2) {
        return std::vector<int>(); // Return an empty vector
    }

    return std::vector<int>(_shape.begin(), _shape.end() - 2);
}

Tensor Tensor::add(const Tensor& other) const {
    if (this->_shape != other._shape) {
        throw std::runtime_error("Shapes must match to perform addition");
    }

    Tensor result(this->_shape);
    for (int i = 0; i<_data.size(); i++) {
        result._data[i] = this->_data[i] + other._data[i];
    }
    return result;
}

Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
    if (a._shape.size() < 2 || b._shape.size() < 2) {
        throw std::runtime_error("This matmul implementation requires at least 2D tensors");
    }

    if (a._shape[a._shape.size()-1] != b._shape[b._shape.size() -2]){
        throw std::runtime_error("Inner dimensions don't match for matmul");
    }
    int l1 = a._shape.size();
    int l2 = b._shape.size();
    int target_rank = std::max(l1, l2);

    //S1. add padding to tensors
    std::vector<int> padded_sa(target_rank,1);
    std::vector<int> padded_sb(target_rank,1);
    int offset_a = target_rank-l1;
    for (int i = 0; i<l1; i++) padded_sa[offset_a+i] = a._shape[i];

    int offset_b = target_rank-l2;
    for (int i = 0; i<l2; i++) padded_sb[offset_b+i] = b._shape[i];

     // output matrix dim = (N, M);
    int M = a._shape[l1-2];
    int K = a._shape[l1-1];
    int N = b._shape[l2-1];
    int batch_dims = target_rank - 2;
    int diff_a = batch_dims - (l1 - 2);
    int diff_b = batch_dims - (l2 - 2);

    // S2. shape of the resultant matrix
    std::vector<int> shape_result(target_rank, 1);
    for (int i = 0; i<padded_sa.size()-2; i++){
        if(padded_sa[i] != padded_sb[i] && padded_sa[i] != 1 && padded_sb[i] != 1){
            throw std::runtime_error("Batch dimensions cant be broadcasted");
        }
        shape_result[i] = std::max(padded_sa[i], padded_sb[i]);
    }
    shape_result[target_rank-2] = M;
    shape_result[target_rank-1] = N;

    Tensor result(shape_result);
    // S3. calculating strides for the batches (till size() -2 )
    std::vector<int> batch_strides_a(target_rank - 2, 0);
    std::vector<int> batch_strides_b(target_rank - 2, 0);

    for (int i = 0; i<(l1-2); i++) batch_strides_a[diff_a+i] = a._strides[i];
    for (int i = 0; i<(l2-2); i++) batch_strides_b[diff_b+i] = b._strides[i];

    for (int i = 0; i < batch_dims; i++) {
        if (padded_sa[i] == 1) batch_strides_a[i] = 0;
        if (padded_sb[i] == 1) batch_strides_b[i] = 0;
    }

    // S4. Count total batch iterations (product of result batch dims)
    int total_batches = 1;
    for (int i = 0; i<batch_dims; i++) total_batches*=shape_result[i];

    // S5. Iterate over each batch
    for (int batch = 0; batch<total_batches; batch++){

        std::vector<int> batch_coord(batch_dims);
        int tmp =batch;
        for (int i = batch_dims-1; i>=0; i--){
            batch_coord[i] = tmp%shape_result[i];
            tmp/=shape_result[i];
        }

        // compute bases
        int base_a = 0, base_b = 0;

        for (int i = 0; i<batch_dims; i++){
            base_a+=batch_coord[i]*batch_strides_a[i];
            base_b+=batch_coord[i]*batch_strides_b[i];
        }

        int base_r = batch*M*N;
        // standard 2d multiplication
        for (int m = 0; m<M; m++){
            for (int n = 0; n<N; n++){
                float sum = 0.0f;
                for(int k = 0; k<K; k++){
                    sum+=a._data[base_a+m*K+k] * b._data[base_b+k*N+n];
                }
                result._data[base_r+m*N+n] = sum;
            }
        }
    }

    return result;

}

// shape shifter functions
Tensor Tensor::Transpose() const{
    // swap the last to dim of shape and strides;
    if (_shape.size() < 2) {
        throw std::runtime_error("Cannot transpose a tensor with less than 2 dimensions.");
    }
    Tensor result = *this;
    int last = result._shape.size()-1;
    int second_last = result._shape.size()-2;
    std::swap(result._shape[last], result._shape[second_last]);
    std::swap(result._strides[last], result._strides[second_last]);
    return result;
}

Tensor Tensor::reshape(const std::vector<int>& newShape) const {
    int totalElements = 1;
    for (int i: newShape) totalElements*=i;

    if (totalElements != this->size()){
        throw std::runtime_error("Reshape failed: Total number of elements must remain the same.");
    }

    Tensor result = *this;
    result._shape = newShape;
    result._strides.resize(newShape.size());
    int multiplier = 1;
    for (int i = newShape.size()-1; i>=0; i--){
        result._strides[i] = multiplier;
        multiplier*=newShape[i];
    }

    return result;
}

 // scalar math
 Tensor Tensor::add(float val) const{
     Tensor result = *this;
     for (float& v: result._data){
         v += val;
     }
     return result;
 }

Tensor Tensor::mul(float val) const{
     Tensor result = *this;
     for (float& v: result._data){
         v *= val;
     }
     return result;
 }

 Tensor Tensor::exp() const{
     Tensor result = *this;
     for (float& v: result._data){
         v = std::exp(v);
     }
     return result;
 }

 Tensor Tensor::log() const{
     Tensor result = *this;
     for (float& v: result._data){
         v = std::log(v);
     }
     return result;
 }

 // Reductions
Tensor Tensor::sum() const{
    Tensor result({1});
    float total = 0.0f;
    for (float v:this->_data){
        total+=v;
    }
    result._data[0] = total;
    return result;
}


Tensor Tensor::sum(int dim) const{
    if (dim>=_shape.size() || dim<0){
        throw std::runtime_error("Dimension cant be greater than shape size");
    }
    std::vector<int> target_shape = this->_shape;
    target_shape[dim] = 1;
    Tensor result(target_shape);
    for (int i = 0; i<_data.size(); i++){
        std::vector<int> coord = get_coordinates(i);
        coord[dim] = 0;
        int result_flat_idx = result.get_flat_index(coord);
        result._data[result_flat_idx] += _data[i];
    }

    return result;
}
