#include <vector>
#include <iostream>
#include <string>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/sleep.hpp"
#include "caffe/util/print.hpp"
#include "caffe/binary.hpp"

extern bool BINARY;
extern bool QUANTIZE;
extern bool DEBUG;
extern bool SCALE_WEIGHTS;
extern bool TERNARY;

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

if(BINARY){
  this->blobs_[0]->binarize_data();
}

if(TERNARY){

// LOG(INFO) << "ternary in innerproduct layer.";
// caffe_sleep(3);

/*
if(DEBUG){
  std::cout << std::endl;
  std::cout << "---------------- Before Ternary --------------------" << std::endl;
  this->blobs_[0]->print_head();
  sleep(3);
}
*/

/*
std::string phase;
if(this->phase_ == TRAIN){
  phase = "train";
}else if(this->phase_ == TEST){
  phase = "test";
}else{
  phase = "run";
}
*/

//LOG(INFO) << "inner product layer: " << phase;
//caffe_sleep(1);

  this->blobs_[0]->ternarize_data(this->phase_);

/*
if(this->phase_ == TEST){
 Dtype alpha = (Dtype) this->blobs_[0]->get_alpha();

for(int i=0; i<bottom.size(); i++){
  Blob<Dtype>* blob = bottom[i];
  caffe_gpu_scale(blob->count(), alpha, blob->gpu_data(), blob->mutable_gpu_data());
}
}
*/

/*
if(DEBUG){
  std::cout << "---------------- After Ternary --------------------" << std::endl;
  this->blobs_[0]->print_head();
  sleep(3);
}
*/

} // end of ternary

if(SCALE_WEIGHTS){
  this->blobs_[0]->acmean_data();
  this->blobs_[0]->scale_binary();
}

  const Dtype* weight = (BINARY || TERNARY) ? this->blobs_[0]->gpu_binary() : this->blobs_[0]->gpu_data();
/*
  const Dtype* weight = NULL;
  if(BINARY || TERNARY) {
	weight = this->blobs_[0]->gpu_binary();

if(DEBUG){
    const Dtype* b = this->blobs_[0]->cpu_binary();
	std::cout << "\nUsing ternarized weight... " << std::endl;
	for(int i=0; i<10; i++){
		std::cout << b[i] << " ";
	}
	std::cout << std::endl;
	// caffe_sleep(1);
}
  }else{
    weight = this->blobs_[0]->gpu_data();
  }
*/

  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();

if(QUANTIZE){
	bottom[0]->quantize_data();
}
	const Dtype* bottom_data = (QUANTIZE) ? bottom[0]->gpu_quantum() : bottom[0]->gpu_data();

/*
if(DEBUG){
	LOG(INFO) << "--------------- bottom data ------------------";
	for(int i=0; i<10; i++){
		std::cout << bottom[0]->cpu_data()[i] << " ";
	}
	std::cout << std::endl;
	LOG(INFO) << "--------------- quantized data ------------------";
	for(int i=0; i<10; i++){
		std::cout << bottom[0]->cpu_quantum()[i] << " ";
	}
	std::cout << std::endl;
	caffe_sleep(3);
}
*/

    // const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }

  const Dtype* weights = (BINARY || TERNARY) ? this->blobs_[0]->gpu_binary() : this->blobs_[0]->gpu_data();
  // const Dtype* weights = this->blobs_[0]->gpu_data();

/*
if(TERNARY){
  Dtype alpha = (Dtype) this->blobs_[0]->get_alpha();

for(int i=0; i<top.size(); i++){
  Blob<Dtype>* blob = top[i];
  caffe_gpu_scale(blob->count(), alpha, blob->gpu_data(), blob->mutable_gpu_data());
}

}
*/
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          // (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)1., top_diff, weights,
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         // (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
         (Dtype)1., top_diff, weights,
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
