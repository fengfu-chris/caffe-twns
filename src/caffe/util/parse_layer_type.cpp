#include "caffe/util/parse_layer_type.hpp"
#include <iostream>

std::string parse_layer_type(std::string layer_name){
  const int n = layer_name.size();
  if(n < 3){
    std::cout << "Invalid layer name. Size must be larger than 3." << std::endl;
  }

  std::string layer_type;
  if((n == 3 || n == 4) && (layer_name.substr(0,2) == "fc" || layer_name.substr(0,2) == "ip")){
    layer_type = "inner_product";
  }else if( n >= 6 && layer_name.substr(0, 6) == "fc_out"){
    layer_type = "inner_product";
  }else if((n == 5 || n == 6) && layer_name.substr(0,4) == "conv"){
    layer_type = "convolution";
  }else if( n >= 5 && layer_name.substr(n-3, 3) == "_bn"){
    layer_type = "batch_norm";
  }else if( n >= 8 && layer_name.substr(n-6, 6) == "_scale"){
    layer_type = "scale";
  }else if( n >= 7 && layer_name.substr(n-5, 5) == "_relu"){
    layer_type = "relu";
  }else if( n >= 4 && layer_name.substr(0, 4) == "loss"){
    layer_type = "loss";
  }else{
    layer_type = "unknown_type";
  }

  return layer_type;
}
