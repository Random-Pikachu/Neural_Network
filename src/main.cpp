#include <iostream>
#include "tensor.h"
using namespace std;

int main() {
    Tensor a({2,2,2});
    Tensor b({2, 2});
    a.load_data({1,2,3,4,5,6,7,8});
    b.load_data({2,0,1,3});

    Tensor c = Tensor::matmul(a, b);

    for (int i = 0; i < c.size(); i++) {
        cout << c.data()[i] << " ";
    }
    cout << endl;
    return 0;
}
