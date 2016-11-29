#include<iostream>

template<typename Dtype>
void print_array(const Dtype* arr, int n){
  int max_n = (n <= 20 ) ? n : 20;
  for(int i=0; i<max_n; i++){
	std::cout << arr[i] << " ";
  }
  std::cout << std::endl;
}
