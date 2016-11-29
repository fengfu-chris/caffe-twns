#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/sleep.hpp"
#include <ctime>
#include <iostream>
#include <sstream>
namespace caffe {
int save_num = 1;
int save_tag = 1;
//cv::RNG rng(uint64( std::time(0) ));

template<typename Dtype>
void DataTransformer<Dtype>::rescale_short_jitter(cv::Mat& img, int short_min, int short_max){

// std::cout << "stepped into rescale_short_jitter(cv::Mat&, int, int)";
//caffe_sleep(2);

    srand((int)time(NULL));
    int scaled_short = (rand() % (short_max - short_min + 1)) + short_min;
	int m = img.rows, n = img.cols;
	int new_m = (m <= n) ? scaled_short : m * double(scaled_short) / double(n);
	int new_n = (m > n)  ? scaled_short : n * double(scaled_short) / double(m);
	resize(img, img, cv::Size(new_n, new_m), 0, 0, cv::INTER_CUBIC);
}

//
template<typename Dtype>
void DataTransformer<Dtype>::light_correction_jitter(cv::Mat& inputImg, float delta)
{
  int imgType = -1;
  if(inputImg.channels() == 3)
    imgType = CV_32FC3;
  else
    imgType = CV_32F;

  cv::Mat tmpMat = cv::Mat::zeros(inputImg.rows, inputImg.cols, imgType);
  inputImg.convertTo(tmpMat, tmpMat.type(), 1.0 / 255.0);
  pow(tmpMat, delta, tmpMat);
  tmpMat.convertTo(inputImg, inputImg.type(), 255.0);
}

template<typename Dtype>
void DataTransformer<Dtype>::rotate_rescale_jitter(cv::Mat& inputImg, float angle, float scale, cv::Mat& rotateMat)
{
  cv::Point2f center = cv::Point2f(inputImg.cols / 2, inputImg.rows / 2); 
  rotateMat = cv::getRotationMatrix2D( center, angle, scale );
  cv::Mat rotateImg;
  cv::warpAffine( inputImg, rotateImg, rotateMat, inputImg.size() );
  inputImg = rotateImg;
}

template<typename Dtype>
void DataTransformer<Dtype>::random_crop_square_jitter(cv::Mat& inputImg, int bordSizeW, int bordSizeH)
{
  int w = inputImg.size().width;
  int h = inputImg.size().height;
  
  cv::Mat tmpImg;
  if(w > h){
    int randW = Rand(w-h);
    tmpImg = inputImg( cv::Rect(randW, 0, h, h) );
  }else if( w < h){
    int randH = Rand(h-w); 
    tmpImg=inputImg( cv::Rect(0, randH, w, w) );
  }else{
    tmpImg = inputImg;
  }
  cv::resize( inputImg, tmpImg, cv::Size(bordSizeW, bordSizeH) );
}

template<typename Dtype>
void DataTransformer<Dtype>::center_crop_square_jitter( cv::Mat& inputImg, int bordSizeW, int bordSizeH)
{
  int w = inputImg.size().width;
  int h = inputImg.size().height;
  
  cv::Mat tmpImg;
  if(w > h){
    int randW = (w-h)/2;
    tmpImg=inputImg(cv::Rect(randW, 0, h, h));
  }else if( w < h){
    int randH = (h-w)/2; 
    tmpImg = inputImg(cv::Rect(0, randH, w, w));
  }else{
    tmpImg = inputImg;
  }
  cv::resize( inputImg, tmpImg, cv::Size(bordSizeW, bordSizeH) );
}


template<typename Dtype>
void DataTransformer<Dtype>::blur_jitter( cv::Mat& inputImg, int kernelSize)
{
  cv::GaussianBlur(inputImg, inputImg, cv::Size(kernelSize, kernelSize), 0);
}

template<typename Dtype>
void DataTransformer<Dtype>::saturate_cast_check( cv::Mat& inputImg ){
  int row_num = inputImg.rows;
  int col_num = inputImg.cols;
  for( int i = 0; i < row_num; ++i ){
    for( int j = 0; j < col_num; ++j ){
      if( inputImg.at<uchar>( i,j ) > 255 )
        inputImg.at<uchar>(i,j) = 255;
      if( inputImg.at<uchar>( i, j ) < 0 )
        inputImg.at<uchar>(i,j) = 0;
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::color_casting( cv::Mat& inputImg ){
  //cv::RNG rng;
  bool alert_r = false;
  bool alert_g = false;
  bool alert_b = false;
  //float r = rng.uniform( 0.f, 1.f );
  //float g = rng.uniform( 0.f, 1.f );
  //float b = rng.uniform( 0.f, 1.f );
  if( Rand(2) == 0 )
    alert_r = true;
  if( Rand(2) == 0 )
    alert_g = true;
  if( Rand(2) == 0 )
    alert_b = true;
  std::vector< cv::Mat > channel_rgb;
  cv::split( inputImg, channel_rgb );
  /*
  if( r > 0.5 )
    alert_r = true;
  if( g > 0.5 )
    alert_g = true;
  if( b > 0.5 )
    alert_b = true;
  */
  if( alert_r == true ){
    bool is_plus = false;
    if( Rand(2) == 0 )
      is_plus = true;
    if( is_plus == true ){
      channel_rgb[0] = channel_rgb[0] + uchar(20);
    }
    else{
      channel_rgb[0] = channel_rgb[0] - uchar(20);
    }
  }

  if( alert_g == true ){
    bool is_plus = false;
    if( Rand(2) == 0 )
      is_plus = true;
    if( is_plus == true ){
      channel_rgb[1] = channel_rgb[1] + uchar(20);
    }
    else{
      channel_rgb[1] = channel_rgb[1] - uchar(20);
    }
  }
  if( alert_b == true ){
    bool is_plus = false;
    if( Rand(2) == 0 )
      is_plus = true;
    if( is_plus == true ){
      channel_rgb[2] = channel_rgb[2] + uchar(20);
    }
    else{
      channel_rgb[2] = channel_rgb[2] - uchar(20);
    }
  }

  saturate_cast_check( channel_rgb[0] );
  saturate_cast_check( channel_rgb[1] );
  saturate_cast_check( channel_rgb[2] );
  cv::merge( channel_rgb, inputImg );
  //inputImg = cv::saturate_cast<uchar>( inputImg );
 
}

template<typename Dtype>
inline float DataTransformer<Dtype>::calcu_dist( cv::Point p1, cv::Point p2 ){
  return sqrt( pow( (float)( p1.x - p2.x ), 2) + pow((float)(p1.y - p2.y), 2) );
}

template<typename Dtype>
inline float DataTransformer<Dtype>::calcu_max_dist( cv::Size img_size, cv::Point center ){
  std::vector< cv::Point >  corners;
  corners.push_back( cv::Point(0,0) );
  corners.push_back( cv::Point( img_size.width , 0 ) );
  corners.push_back( cv::Point( 0, img_size.height ) );
  corners.push_back( cv::Point( img_size.width, img_size.height ) );

  float max_dist = 0.;
  for( int i = 0; i < corners.size(); ++i ){
    float dist_tmp = calcu_dist( corners[i], center );
    if( dist_tmp > max_dist )
      max_dist = dist_tmp; 
  }
  
  return max_dist;
}

template<typename Dtype>
void DataTransformer<Dtype>::generate_mask_for_vignet( cv::Mat& mask_img, float radius, float mask_power ){
  cv::Point center_point = cv::Point( mask_img.size().width/2, mask_img.size().height/2 );
  float img_max_radius = calcu_max_dist( mask_img.size(), center_point );
  
  mask_img.setTo( cv::Scalar(1) );
  
  for( int i = 0; i < mask_img.rows; ++i ){
    for( int j = 0; j < mask_img.cols; ++j ){
      float dist_tmp = calcu_dist( center_point, cv::Point( j, i ) )/img_max_radius;
      dist_tmp *= mask_power;
      float dist_result = pow( cos(dist_tmp), 4 );
      mask_img.at<float>(i,j) = dist_result;    
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::vignetting( cv::Mat& input_img, const float radius, const float mask_power ){
  cv::Mat mask_img( input_img.size(), CV_32F );
  generate_mask_for_vignet( mask_img, radius, mask_power );
 
  cv::Mat lab_img( input_img.size(), CV_8UC3 );
  cv::cvtColor( input_img, lab_img, CV_BGR2Lab );
  for( int row = 0; row < lab_img.rows; ++row ){
    for( int col = 0; col < lab_img.cols; ++ col ){
      cv::Vec3b val_tmp = lab_img.at< cv::Vec3b >( row, col );
      val_tmp.val[0] *= mask_img.at<float>( row, col );
      lab_img.at< cv::Vec3b >(row, col) = val_tmp;
    }
  }
  cv::cvtColor( lab_img, input_img, CV_Lab2BGR );
}

template<typename Dtype>
float DataTransformer<Dtype>::calcu_shift( float x1, float x2, float cx, float k ){
  float thresh = 1.;
  float x3 = x1 + ( x2 - x1 )*0.5;
  float res1 = x1 + ((x1 - cx)*k*((x1 - cx)*(x1 - cx)));
  float res3 = x3 + ((x3 - cx)*k*((x3 - cx)*(x3 - cx)));

  if( res1 > -thresh && res1 < thresh )
    return x1;
  if( res3 < 0. )
    return calcu_shift( x3, x2, cx, k );
  else
    return calcu_shift( x1, x3, cx, k );
}

template<typename Dtype>
float DataTransformer<Dtype>::get_radial_X( float x, float y, float cx, float cy, float k, bool scale, cv::Vec4f props ){
  float result;
  if( scale ){
    float x_shift = props[0];
    float y_shift = props[1];
    float x_scale = props[2];
    float y_scale = props[3];

    x = x*x_scale + x_shift;
    y = y*y_scale + y_shift;
    result = x + ( (x - cx)*k*((x - cx)*(x - cx) + (y - cy)*(y - cy)) );
  }
  else
    result = x + ( (x - cx)*k*((x - cx)*(x - cx) + (y - cy)*(y - cy)));

  return result;
  
}

template<typename Dtype>
float DataTransformer<Dtype>::get_radial_Y( float x, float y, float cx, float cy, float k, bool scale, cv::Vec4f props){
  float result = 0.;
  if( scale ){
    float x_shift = props[0];
    float y_shift = props[1];
    float x_scale = props[2];
    float y_scale = props[3];

    x = x*x_scale + x_shift;
    y = y*y_scale + y_shift;
    
    result = y + ((y - cy)*k*((x - cx)*(x - cx) + (y - cy)*(y - cy)));
  } 
  else
    //result = y + ((y - cy)*k*((x - cx)*(y - cy)));
    result = y + ((y - cy)*k*((x - cx)*(x - cx) + (y - cy)*(y - cy)));

  return result;
}

//Cx Cy specify the coordinates from where the distorted image will have as initial point.
//k specifies the distortion factor;
template<typename Dtype>
void DataTransformer<Dtype>::fish_eye_distortion( cv::Mat& input_img, float Cx, float Cy, float k, bool scale){

  CHECK( Cx >= 0 );
  CHECK( Cy >= 0 );
  CHECK( k >= 0 );

  cv::Mat map_x = cv::Mat( input_img.size(), CV_32FC1 );
  cv::Mat map_y = cv::Mat( input_img.size(), CV_32FC1 );
  
  int rows = input_img.rows;
  int cols = input_img.cols;

  cv::Vec4f props;

  float x_shift = calcu_shift( 0, Cx - 1, Cx, k );
  props[0] = x_shift;
  float new_center_x = cols - Cx;
  float x_shift_2 = calcu_shift( 0, new_center_x - 1, new_center_x, k );
 
  float y_shift = calcu_shift( 0, Cy - 1, Cy, k );
  props[1] = y_shift;
  float new_center_y = cols - Cy;
  float y_shift_2 = calcu_shift( 0, new_center_y - 1, new_center_y, k );

  float x_scale = ( cols - x_shift - x_shift_2 )/cols;
  props[2] = x_scale;
  float y_scale = ( rows - y_shift - y_shift_2 )/rows;
  props[3] = y_scale;

  float* p = map_x.ptr<float>(0);
  for( int row = 0; row < rows; ++row ){
    for( int col = 0; col < cols; ++col ){
      *p++ = get_radial_X( (float)col, (float)row, Cx, Cy, k, scale, props );
    }
  }
  
  p = map_y.ptr<float>(0);
  for( int row = 0; row < rows; ++row ){
    for( int col = 0; col < cols; ++ col ){
      *p++ = get_radial_Y( (float)col, (float)row, Cx, Cy, k, scale, props );
    }
  }

  cv::Mat output_img;
  cv::remap( input_img, output_img, map_x, map_y, CV_INTER_LINEAR, cv::BORDER_CONSTANT );
  output_img.copyTo( input_img ); 
}


template<typename Dtype>
void DataTransformer<Dtype>::jitter_image_total( cv::Mat& cv_img_, const int crop_size ){
    //std::cout << "begin to jitter the image!!!!!!!!!!!!!!!!!!!!!!\n";
    //int save_tag = Rand(10000);
    std::stringstream ss;
    ss << save_tag;
    std::string save_directory = "/mnt/lvm/heyuhang/train_test_0314/test_img/";
    std::string save_name = save_directory + "_before_jitter_" + ss.str() + ".jpg";
    //cv::imwrite( save_name, cv_img_ );

    int jitter_num = 0;
   
    float resize_scale_ratio = param_.resize_scale_ratio();
    int ecrop_size = int(crop_size*resize_scale_ratio);

    // step 1 : rotate & rescale
    int max_rotate_degree = param_.max_rotate_degree();
    float max_rescale_ratio = param_.max_rescale_ratio();
    if( phase_ == TRAIN && Rand(2) == 0)
    {
      jitter_num++;
      int sign = Rand(2) == 0?1:-1;
      float degree = max_rotate_degree > 0 ? Rand(max_rotate_degree)*sign : 0;
      sign = Rand(2) == 0?1:-1;  
      float scale = max_rescale_ratio > 0 ? 1.0 + Rand(int(max_rescale_ratio*100))/100.0*sign : 1.0;
      cv::Mat rotMat;
      rotate_rescale_jitter(cv_img_,degree, scale, rotMat);
    }
    // step 2 : gaussian blur
    if(phase_ == TRAIN && Rand(2) == 0 && param_.blur_jitter()) 
    {
      jitter_num++;
      int ks = ( Rand(2) == 0?3:5 ); 
      blur_jitter(cv_img_, ks);
    }
    // step 3 : light jitter
    int max_gamma_light = param_.max_gamma_light();
    int base_gamma_light = param_.base_gamma_light();
    if(phase_ == TRAIN && Rand(2) == 0 && max_gamma_light > 0 && base_gamma_light > 0) 
    {
      jitter_num++;
      int seed = Rand(max_gamma_light); 
      Dtype delta = 0.1*(seed + base_gamma_light); //0.5-2.0 
      light_correction_jitter(cv_img_, delta);
    } 
    // step 4 : crop square --> just resize @ here
  
    if(phase_ == TRAIN) 
      random_crop_square_jitter(cv_img_, ecrop_size, ecrop_size);
    else
      center_crop_square_jitter(cv_img_, ecrop_size, ecrop_size);

    // step5: color casting
    if( phase_ == TRAIN && Rand(2) == 0 ){
      jitter_num++;
      color_casting( cv_img_ );
    }
    //step 6: vignetting
    float radius_vignet = param_.vignet_radius();
    float vignet_mask_power = param_.vignet_mask_power();
    if( phase_ == TRAIN && Rand(2) == 0 ){
      jitter_num++;
      vignetting( cv_img_, radius_vignet, vignet_mask_power );
     }
    
    //step 7: fish_eye_distortion
    float distortion_factor = param_.distortion_factor();
    if( phase_ == TRAIN && Rand(2) == 0 ){
      jitter_num++;
      fish_eye_distortion( cv_img_, cv_img_.cols/2.0, cv_img_.rows/2.0, distortion_factor, true );
    } 

    if( phase_ == TRAIN ){  
      //std::cout << "jitter_num = " << jitter_num << std::endl << std::endl;      
      //save_name = save_directory + "_jittered_" + ss.str() + ".jpg";
      //if( save_num < 100 )
        //cv::imwrite( save_name, cv_img_ );
      //save_tag++;
      save_num++;
    }
}
// util end

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param, Phase phase) : param_(param), phase_(phase) {
  // check if we want to use mean_file
  // std::cout << "stepped into DataTransformer(const TransformationParameter&, Phase)" << std::endl;
  //
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {

 // std::cout << "stepped into Transform(const Datum&, Dtype* )" << std::endl;
  //caffe_sleep(2);

  const int crop_size = param_.crop_size();
  //std::cout << "Entered the Transform Function!\n";
  //std::cout << "image channels = " << datum.channels() << std::endl; 
  Datum datum_;
  datum_ = datum;

  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  //const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
   //std::cout << "Entered Transform(Datum, Blob<Dtype>* transformed_blob) function!\n";
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {

// std::cout << "datum.encoded() = true" << std::endl;

#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";

    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      printf("use the DecodeDatumToCVMat function!\n");
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      printf("use the DecodeDatumToCVMatNative function!\n");
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    return Transform(cv_img, transformed_blob);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
// std::cout << "datum.encoded() = false" << std::endl;
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  // std::cout << "entered this function without OpenCV!\n";
  Transform(datum, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img_,
                                       Blob<Dtype>* transformed_blob) {
  cv::Mat cv_img;
  cv_img_.copyTo( cv_img );

  // std::cout << "stepped into transform(const cv::Mat&, Blob<Dtype>*)" << std::cout;
  // caffe_sleep(2);
  // rescale_short_jitter(cv_img);

  // jitter_image_total( cv_img, param_.crop_size() );

  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
    //cv::imwrite( save_name, cv_cropped_img );
    //save_tag++;
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}
#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_size) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_size, crop_size);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }
  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_size)? crop_size: datum_height;
  shape[3] = (crop_size)? crop_size: datum_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_size)? crop_size: img_height;
  shape[3] = (crop_size)? crop_size: img_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN && param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
