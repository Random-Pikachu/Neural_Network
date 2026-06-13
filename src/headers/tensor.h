 // #pragma  once
 #ifndef  TENSOR_H
 #define TENSOR_H
 #include <vector>

 class Tensor
 {
     private:
        std::vector<int> _shape;  // (1,1,1,1) -> 4d tensor
        // {1.0, 2.0, 3.0, 4.0, 5.0, ...} store everything in a flat vector and then calculate flat index of desired (r,c) combo
        std::vector<float> _data;
        std::vector<int> _strides; //multipliers for each dimension

     public:
         Tensor(std::vector<int> _shape);
         void load_data(const std::vector<float> &input_data);

         int get_flat_index(const std::vector<int> &coordinates) const;
         std::vector<int> get_batch_shape() const;
         Tensor stride(const Tensor& a);
         
         // accessors
         const std::vector<int>& shape() const;
         const std::vector<float>& data() const;
         std::vector<float>& data();
         int size() const;

         float at(const std::vector<int> &coordinates) const;
         void set(const std::vector<int> &coordinates, float value);

         Tensor add(const Tensor& other) const;
         static Tensor matmul(const Tensor& a, const Tensor& b);
 };

 #endif