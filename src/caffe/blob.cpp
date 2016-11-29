#include <climits>
#include <vector>
#include <iostream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/sleep.hpp"
#include "caffe/util/print.hpp"
#include "caffe/binary.hpp"

extern bool DEBUG;
extern bool QUANTIZE;
extern int TERNARY_DELTA;

namespace caffe {

// Implemented @ 2016-3-9
// Deprecated
/*
template <typename Dtype>
void Blob<Dtype>::binary(){
  Dtype* blob = this->mutable_cpu_data();
  for(int i=0; i<this->count(); i++){
	blob[i] = (blob[i] >= 0) ? 1 : -1;
  }
}
*/

// Implemented @ 2016-3-21
template <typename Dtype>
vector<int> Blob<Dtype>::ternary_stats(){
	if(ternary_stats_ == NULL){
		ternary_stats_ = new vector<int>(3,0);
	}
	CHECK_EQ((*ternary_stats_).size(), 3);
	for(int i=0; i<3; i++){
		(*ternary_stats_)[i] = 0;
	}
	float eps = 1e-6;
	const Dtype* X = this->cpu_binary();
	for(int i=0; i<this->count(); i++){
		Dtype x = X[i];
		int id = 2 * (x>eps) + 0 * (x<-eps) + 1*(x>=-eps && x<eps);
		(*ternary_stats_)[id] += 1;
	}

	return *ternary_stats_;
}

// Implemented @ 2016-3-14
template <typename Dtype>
void Blob<Dtype>::clip_data(){
    caffe_gpu_clip(count_, mutable_gpu_data());
/*
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
	caffe_clip(count_, mutable_cpu_data());
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    caffe_gpu_clip(count_, mutable_gpu_data());
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
*/
}

// Implemented @ 2016-3-13
/*
template <typename Dtype>
void Blob<Dtype>::binarize_data(const Dtype bounds){
   caffe_gpu_sign<Dtype>(this->count(), this->gpu_data(), this->mutable_gpu_binary());
   caffe_gpu_scal<Dtype>(this->count(), bounds, this->mutable_gpu_binary());
}
*/

// revised 2016-3-16
template <typename Dtype>
void Blob<Dtype>::binarize_data(const Dtype bounds){
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
	caffe_cpu_sign<Dtype>(this->count(), this->cpu_data(), this->mutable_cpu_binary());
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
   caffe_gpu_sign<Dtype>(this->count(), this->gpu_data(), this->mutable_gpu_binary());
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

// revised 2016-4-14
template <typename Dtype>
void Blob<Dtype>::set_delta(){
  float scale_factor = TERNARY_DELTA * 1.0 / 10; 
  Dtype delta = (Dtype) scale_factor * this->asum_data() / this->count();
  delta = (delta <= 100) ? delta : 100;
  delta = (delta >= -100) ? delta : -100; 
  this->delta_ = delta;
}

template <typename Dtype>
void Blob<Dtype>::set_delta(Dtype delta){
  delta = (delta <= 100) ? delta : 100;
  delta = (delta >= -100) ? delta : -100;
  this->delta_ = delta;
}

template <typename Dtype>
Dtype Blob<Dtype>::get_delta() const{
  return this->delta_;
}

template <typename Dtype>
void Blob<Dtype>::set_alpha(Dtype alpha){
  alpha = (alpha <= 100) ? alpha : 100;
  alpha = (alpha >= -100) ? alpha : -100;
  this->alpha_ = alpha;
}

template <typename Dtype>
Dtype Blob<Dtype>::get_alpha() const{
  return this->alpha_;
}

// revised 2016-3-21
template <typename Dtype>
void Blob<Dtype>::ternarize_data(Phase phase){

if(phase == RUN){

// if(DEBUG) print_head();

 //LOG(INFO) << "RUN phase...";
 // caffe_sleep(3);
 return; // do nothing for the running phase
}else if(phase == TRAIN){
 //LOG(INFO) << "TRAIN phase ...";
 // caffe_sleep(3);
}else{
 //LOG(INFO) << "TEST phase ...";
 // caffe_sleep(3);
}

  // const Dtype delta = 0; // default value; 
  // const Dtype delta = (Dtype) 0.8 * this->asum_data() / this->count();
  this->set_delta();
  const Dtype delta = this->get_delta();
  Dtype alpha = 1;

  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
{
	caffe_cpu_ternary<Dtype>(this->count(), delta, this->cpu_data(), this->mutable_cpu_binary());
	alpha = caffe_cpu_dot(this->count(), this->cpu_binary(), this->cpu_data());
	alpha /= caffe_cpu_dot(this->count(), this->cpu_binary(), this->cpu_binary());
	caffe_cpu_scale(this->count(), alpha, this->cpu_binary(), this->mutable_cpu_binary());
	// this->set_alpha(alpha);
}
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
{
    caffe_gpu_ternary<Dtype>(this->count(), delta, this->gpu_data(), this->mutable_gpu_binary());
	Dtype* pa = new Dtype(0);
	caffe_gpu_dot(this->count(), this->gpu_binary(), this->gpu_data(), pa);
	Dtype* pb = new Dtype(0);
	caffe_gpu_dot(this->count(), this->gpu_binary(), this->gpu_binary(), pb);
	
	alpha = (*pa) / ((*pb) + 1e-6);
	this->set_alpha(alpha);

	caffe_gpu_scale(this->count(), alpha, this->gpu_binary(), this->mutable_gpu_binary());
	// this->set_alpha((Dtype)1);

    // LOG(INFO) << "alpha = " << alpha;
	// caffe_sleep(3);
}
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

// Implemented 2016-3-16
template <typename Dtype>
void Blob<Dtype>::quantize_data(const Dtype left, const Dtype right){
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
// cpu codes
if(DEBUG){
	LOG(INFO) << "CPU codes.";
	caffe_sleep(3);
}
    caffe_quantize(this->count(), left, right, this->cpu_data(), this->mutable_cpu_quantum());
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
// gpu codes
if(DEBUG){
	LOG(INFO) << "GPU codes.";
	caffe_sleep(3);
}
    caffe_gpu_quantize(this->count(), left, right, this->gpu_data(), this->mutable_gpu_quantum());

	// DEBUG caffe_gpu_quantize
if(DEBUG){
	std::cout << "--------------- data ----------------" << std::endl;
	for(int i=0; i<10; i++){
		std::cout << this->cpu_data()[i] << " ";
	}
	std::cout << std::endl;

	std::cout << "-------------- quantum ----------------" << std::endl;
	for(int i=0; i<10; i++){
		std::cout << this->cpu_quantum()[i] << " ";
	}
	std::cout << std::endl;

	caffe_sleep(10);
}

    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

// Implemented @ 2016-3-18
template <typename Dtype>
void Blob<Dtype>::acmean_data(){
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
// CPU codes
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
// GPU codes
	Dtype* X;
	X = mutable_cpu_acmean();
/*
	LOG(INFO) << "shape.size() = " << shape().size();
	for(int i=0; i<shape().size(); i++){
		LOG(INFO) << "shape(" << i <<") = " << shape(i);
	}
	LOG(INFO) << "acmean_len = " << acmean_len;
*/
	for(int i=0; i<shape(0); i++){
		const int n = count() / shape(0);
		// Dtype x = 0;
		// caffe_gpu_asum(n, this->gpu_data() + offset(i), &X[i]);
		// NOTE: cann't take this form of operation on GPU!
		X[i] = caffe_cpu_asum(n, this->cpu_data() + offset(i)) / n;
		// caffe_gpu_asum(n, this->gpu_data() + offset(i), &x);
		// X[i] /= n;
	}
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

// Implemented @ 2016-3-18
template <typename Dtype>
void Blob<Dtype>::scale_binary(){
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
// CPU codes
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
// GPU codes
/*
	for(int i=0; i<shape(0); i++){
		const int n = count() / shape(0);
		caffe_gpu_scal<Dtype>(n, this->gpu_acmean()[i], this->mutable_gpu_binary() + offset(i));
	}
*/
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

template <typename Dtype>
void print_forward(const Dtype* p_data, int nums){
  for(int i=0; i<nums; i++){
    std::cout << p_data[i] << " ";
  }
  std::cout << std::endl;
}

template <typename Dtype>
void Blob<Dtype>::print_head(int nums){
  int n = (this->count() <= nums) ? this->count() : nums;

  std::cout << "--- data_ heads ---" << std::endl;
  print_forward<Dtype>(this->cpu_data(), n);
  
  std::cout << "--- diff_ heads ---" << std::endl;
  print_forward<Dtype>(this->cpu_diff(), n);

  std::cout << "--- binary_ heads ---" << std::endl;
  print_forward<Dtype>(this->cpu_binary(), n);

  // std::cout << "--- quantum_ heads ---" << std::endl;
  // print_forward<Dtype>(this->cpu_quantum(), n);
}

template <typename Dtype>
void print_reverse(const Dtype* p_data, int total_nums, int n){
  for(int i=0; i<n; i++){
    std::cout << p_data[total_nums-1-i] << " ";
  }
  std::cout << std::endl;
}

template <typename Dtype>
void Blob<Dtype>::print_tail(int nums){
  int total_nums = this->count();
  int n = (total_nums <= nums) ? total_nums : nums;

  std::cout << "--- data_ tails ---" << std::endl;
  print_reverse<Dtype>(this->cpu_data(), total_nums, n);

  std::cout << "--- diff_ tails ---" << std::endl;
  print_reverse<Dtype>(this->cpu_diff(), total_nums, n);

  std::cout << "--- binary_ tails ---" << std::endl;
  print_reverse<Dtype>(this->cpu_binary(), total_nums, n);

  std::cout << "--- quantum_ tails ---" << std::endl;
  print_reverse<Dtype>(this->cpu_quantum(), total_nums, n);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  // LOG(INFO) << "shape.size() = " << shape.size();
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    count_ *= shape[i];
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
    // LOG(INFO) << " shape(" << i << ")=" << shape[i];
  }
  // caffe_sleep(2);
  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
	binary_.reset(new SyncedMemory(capacity_ * sizeof(Dtype))); // added item @ 2016-3-13

// Save memory
if(QUANTIZE){
	// LOG(INFO) << "In Reshape.";
	// caffe_sleep(3);
	quantum_.reset(new SyncedMemory(capacity_ * sizeof(Dtype))); // added item @ 2016-3-17
}
	// LOG(INFO) << "count = " << count_ << ", capacity = " << capacity_;
	// LOG(INFO) << "Before reset.";
	// LOG(INFO) << "shape[0] = " << shape[0];
	acmean_len = (shape.size() > 0) ? shape[0] : 1;
	acmean_.reset(new SyncedMemory(acmean_len * sizeof(Dtype))); // added item @ 2016-3-18
	// LOG(INFO) << "After reset.";
	// LOG(INFO) << "------------------";
  }
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.shape());
}

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
    const int width)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(num, channels, height, width);
}

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(shape);
}

template <typename Dtype>
const int* Blob<Dtype>::gpu_shape() const {
  CHECK(shape_data_);
  return (const int*)shape_data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

// added term @ 2016-3-13
template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_binary() const {
  CHECK(binary_);
  return (const Dtype*)binary_->cpu_data();
}

// added term @ 2016-3-17
template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_quantum() const {
  CHECK(quantum_);
  return (const Dtype*)quantum_->cpu_data();
}

// added term @ 2016-3-18
template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_acmean() const {
  CHECK(acmean_);
  return (const Dtype*)acmean_->cpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

// added term @ 2016-3-13
template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_binary() const {
  CHECK(binary_);
  return (const Dtype*)binary_->gpu_data();
}

// added term @ 2016-3-17
template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_quantum() const {
  CHECK(quantum_);
  return (const Dtype*)quantum_->gpu_data();
}

// added term @ 2016-3-18
template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_acmean() const {
  CHECK(acmean_);
  return (const Dtype*)acmean_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

// added term @ 2016-3-13
template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_binary() {
  CHECK(binary_);
  return static_cast<Dtype*>(binary_->mutable_cpu_data());
}

// added term @ 2016-3-17
template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_quantum() {
  CHECK(quantum_);
  return static_cast<Dtype*>(quantum_->mutable_cpu_data());
}

// added term @ 2016-3-18
template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_acmean() {
  CHECK(acmean_);
  return static_cast<Dtype*>(acmean_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

// added term @ 2016-3-13
template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_binary() {
  CHECK(binary_);
  return static_cast<Dtype*>(binary_->mutable_gpu_data());
}

// added term @ 2016-3-17
template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_quantum() {
  CHECK(quantum_);
  return static_cast<Dtype*>(quantum_->mutable_gpu_data());
}

// added term @ 2016-3-18
template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_acmean() {
  CHECK(acmean_);
  return static_cast<Dtype*>(acmean_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }

template <typename Dtype>
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    caffe_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->gpu_data()),
        static_cast<Dtype*>(data_->mutable_gpu_data()));
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template <> unsigned int Blob<unsigned int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_data());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_data(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_diff());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_diff(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
  Dtype sumsq;
  const Dtype* data;
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = cpu_data();
    sumsq = caffe_cpu_dot(count_, data, data);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = gpu_data();
    caffe_gpu_dot(count_, data, data, &sumsq);
#else
    NO_GPU;
#endif
    break;
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> unsigned int Blob<unsigned int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
  Dtype sumsq;
  const Dtype* diff;
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = cpu_diff();
    sumsq = caffe_cpu_dot(count_, diff, diff);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = gpu_diff();
    caffe_gpu_dot(count_, diff, diff, &sumsq);
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_data(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
  Dtype* data;
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = mutable_cpu_data();
    caffe_scal(count_, scale_factor, data);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = mutable_gpu_data();
    caffe_gpu_scal(count_, scale_factor, data);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_diff(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
  Dtype* diff;
  if (!diff_) { return; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = mutable_cpu_diff();
    caffe_scal(count_, scale_factor, diff);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = mutable_gpu_diff();
    caffe_gpu_scal(count_, scale_factor, diff);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
}

template <typename Dtype>
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() ||
      other.has_height() || other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 &&
           LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  switch (Caffe::mode()) {
  case Caffe::GPU:
    if (copy_diff) {
      caffe_copy(count_, source.gpu_diff(),
          static_cast<Dtype*>(diff_->mutable_gpu_data()));
    } else {
      caffe_copy(count_, source.gpu_data(),
          static_cast<Dtype*>(data_->mutable_gpu_data()));
    }
    break;
  case Caffe::CPU:
    if (copy_diff) {
      caffe_copy(count_, source.cpu_diff(),
          static_cast<Dtype*>(diff_->mutable_cpu_data()));
    } else {
      caffe_copy(count_, source.cpu_data(),
          static_cast<Dtype*>(data_->mutable_cpu_data()));
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data
  Dtype* data_vec = mutable_cpu_data();
  if (proto.double_data_size() > 0) {
    CHECK_EQ(count_, proto.double_data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.double_data(i);
    }
  } else {
    CHECK_EQ(count_, proto.data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.data(i);
    }

if(DEBUG){
    int n = (count_ <= 20) ? count_ : 20;
    // if(n==0) LOG(INFO) << "empty blob.";
    LOG(INFO) << "-- heads --";
    for(int i=0; i<n; i++) std::cout << data_vec[i] << " ";
    std::cout << std::endl;

    LOG(INFO) << "-- tails --";
    for(int i=0; i<n; i++) std::cout << data_vec[count_-1-i] << " ";
    std::cout << std::endl;
    std::cout << std::endl;
}
  }

  if (proto.double_diff_size() > 0) {
    CHECK_EQ(count_, proto.double_diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.double_diff(i);
    }
  } else if (proto.diff_size() > 0) {
    CHECK_EQ(count_, proto.diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
}

template <typename Dtype>
void Blob<Dtype>::BinaryFromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data
  Dtype* data_vec = mutable_cpu_binary();
  // Dtype* data_vec = mutable_cpu_data();
  if(proto.binary_data_size() > 0){
    CHECK_EQ((count_-1)/32 + 1, proto.binary_data_size());
    for(int i=0; i<proto.binary_data_size(); i++){
	  int n = proto.binary_data(i);
	  for(int j=0; j<32 && i*32+j<count_; j++){
		int b = n & (1 << j);
	    data_vec[i*32+j] = (b != 0) ? 1 : -1;
	  }
    }
  }
}

template <typename Dtype>
void Blob<Dtype>::TernaryFromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }

  this->set_delta((Dtype) proto.delta());
  this->set_alpha((Dtype) proto.alpha());

  // copy data
  // std::cout << "----------- n ----------" << std::endl;
  Dtype* data_vec = mutable_cpu_binary();
  if(proto.ternary_data_size() > 0){
    CHECK_EQ((count_-1)/16+1, proto.ternary_data_size());
    for(int i=0; i<proto.ternary_data_size(); i++){
	  int n = proto.ternary_data(i);
      //if(i<20){
//		std::cout << n << " ";
 //     }
	  for(int j=0; j<16 && i*16+j<count_; j++){
		int b1 = n & (1 << (2*j));
        int b2 = n & (1 << (2*j+1));
		Dtype b = (Dtype)0;
		if(b1 != 0 && b2 != 0){
			b = (Dtype)1;
		}else if(b1 ==0 && b2 ==0 ){
			b = (Dtype)-1;
		}
	    data_vec[i*16+j] = b * this->get_alpha();
	  }
    }
  }
  //std::cout << std::endl;

  //LOG(INFO) << "---------------------";
if(DEBUG){
  LOG(INFO) << "delta = " << this->get_delta() ;
  LOG(INFO) << "alpha = " << this->get_alpha() ;
  int n = (count_ <= 20) ? count_ : 20;

  LOG(INFO) << "-- heads --";
  for(int i=0; i<n; i++)  std::cout << data_vec[i] << " ";
  std::cout << std::endl;
  LOG(INFO) << "-- tails --";
  for(int i=0; i<n; i++)  std::cout << data_vec[count_-1-i] << " ";
  std::cout << std::endl;
  std::cout << std::endl;
}
}


template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  const double* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_double_data(data_vec[i]);
  }
  if (write_diff) {
    const double* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const float* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }

if(DEBUG){
  int n = (count_ <= 20) ? count_ : 20;
  LOG(INFO) << "-- heads --";
  for(int i=0; i<n; i++) std::cout << data_vec[i] << " ";
  std::cout << std::endl;

  LOG(INFO) << "-- tails --";
  for(int i=0; i<n; i++) std::cout << data_vec[count_-1-i] << " ";
  std::cout << std::endl;
  std::cout << std::endl;
}

  if (write_diff) {
    const float* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<float>::BinaryToProto(BlobProto* proto) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_binary_data();
  const float* data_vec = cpu_binary();
  for (int i = 0; i < count_; i += 32) {
	unsigned int n = 0;
	for(int j=0; j<32 && (i+j)<count_ ; j++){
		int b = data_vec[i+j] >= 0 ? 1 : 0;
		n ^= (b << j);
	}
    proto->add_binary_data(n);
  }
}

template <>
void Blob<double>::BinaryToProto(BlobProto* proto) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_binary_data();
  const double* data_vec = cpu_binary();
  for (int i = 0; i < count_; i += 32) {
	unsigned int n = 0;
	for(int j=0; j<32 && (i+j)<count_ ; j++){
		int b = data_vec[i+j] >= 0 ? 1 : 0;
		n ^= (b << j);
	}
    proto->add_binary_data(n);
  }
}

template <>
void Blob<float>::TernaryToProto(BlobProto* proto) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_ternary_data();

// !!! note: should first ternarize data!
//  this->ternarize_data();
//  this->set_delta();
//
  //float delta = (float)this->get_delta();
  const float delta = this->get_delta();
  // proto->clear_delta();
  proto->set_delta(delta);

  //float alpha = (float)this->get_alpha();
  const float alpha = this->get_alpha();
  // proto->clear_alpha();
  proto->set_alpha(alpha);

  // std::cout << "-----------  n  -------------- " << std::endl;
  const float* data_vec = cpu_binary();
  for (int i = 0; i < count_; i += 16) {
	unsigned int n = 0;
	for(int j=0; j<16 && (i+j)<count_; j++){
	  int b1 = data_vec[i+j] > -0.5*alpha ? 1 : 0;
      int b2 = data_vec[i+j] >  0.5*alpha ? 1 : 0;
	  n ^= (b1 << (2*j));
	  n ^= (b2 << (2*j+1));
    }
    proto->add_ternary_data(n);
    // if(i < 16*20	std::cout << n << " ";
  }
  // std::cout << std::endl;

if(DEBUG){
  LOG(INFO) << "delta = " << delta ;
  LOG(INFO) << "alpha = " << alpha ;
  int n = (count_ <= 20) ? count_ : 20;
  LOG(INFO) << "-- heads --";
  for(int i=0; i<n; i++){
    std::cout << data_vec[i] << " ";
  }
  std::cout << std::endl;
  LOG(INFO) << "-- tails --";
  for(int i=0; i<n; i++){
    std::cout << data_vec[count_-1-i] << " ";
  }
  std::cout << std::endl;
  std::cout << std::endl;
}

}

template <>
void Blob<double>::TernaryToProto(BlobProto* proto) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_ternary_data();

// !!! note: should first ternarize data!
  //this->ternarize_data();
//  this->get_delta();
  //float delta = (float)this->get_delta();
  const float delta = this->get_delta();
  // proto->clear_delta();
  proto->set_delta(delta);

  //float alpha = (float)this->get_alpha();
  const float alpha = this->get_alpha();
  // proto->clear_alpha();
  proto->set_alpha(alpha);

  const double* data_vec = cpu_binary();
  for (int i = 0; i < count_; i += 16) {
	unsigned int n = 0;
	for(int j=0; j<16 && (i+j)<count_; j++){
	  int b1 = data_vec[i+j] > -0.5*alpha ? 1 : 0;
      int b2 = data_vec[i+j] >  0.5*alpha ? 1 : 0;
	  n ^= (b1 << (2*j));
	  n ^= (b2 << (2*j+1));
    }
    proto->add_ternary_data(n);
  }
}

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace caffe
