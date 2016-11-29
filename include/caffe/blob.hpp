#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

const int kMaxBlobAxes = 32;

namespace caffe {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Blob {
 public:
  Blob()
       : data_(), diff_(), count_(0), capacity_(0) {}

  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  explicit Blob(const int num, const int channels, const int height,
      const int width);
  explicit Blob(const vector<int>& shape);

  /// Added by lifengfu @ 2016-3-9
  void print_head(int nums = 10);
  void print_tail(int nums = 10);
  void print(int nums = 10){ print_head(nums); }

  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  void Reshape(const int num, const int channels, const int height,
      const int width);
  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  void Reshape(const vector<int>& shape);
  void Reshape(const BlobShape& shape);
  void ReshapeLike(const Blob& other);
  inline string shape_string() const {
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }
  inline const vector<int>& shape() const { return shape_; }
  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }
  inline int num_axes() const { return shape_.size(); }
  inline int count() const { return count_; }

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  inline int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }
  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  inline int count(int start_axis) const {
    return count(start_axis, num_axes());
  }

  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  inline int CanonicalAxisIndex(int axis_index) const {
    CHECK_GE(axis_index, -num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }

  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  inline int num() const { return LegacyShape(0); }
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  inline int channels() const { return LegacyShape(1); }
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  inline int height() const { return LegacyShape(2); }
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  inline int width() const { return LegacyShape(3); }
  inline int LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4)
        << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);
  }

  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num());
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;
  }

  inline int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }
  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_data()[offset(n, c, h, w)];
  }

  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_diff()[offset(n, c, h, w)];
  }

  inline Dtype data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }

  inline Dtype diff_at(const vector<int>& index) const {
    return cpu_diff()[offset(index)];
  }

  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }

  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }

  inline const shared_ptr<SyncedMemory>& binary() const {
	CHECK(binary_);
	return binary_;  
  }

  inline const shared_ptr<SyncedMemory>& quantum() const {
	CHECK(quantum_);
	return quantum_;
  }

  const Dtype* cpu_data() const;
  const Dtype* cpu_diff() const;
  const Dtype* cpu_binary() const;  // added term
  const Dtype* cpu_quantum() const; // added term
  const Dtype* cpu_acmean() const; // added term
  Dtype* mutable_cpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_cpu_binary();  // added term
  Dtype* mutable_cpu_quantum();  // added term
  Dtype* mutable_cpu_acmean();  // added term
  void set_cpu_data(Dtype* data);

  const Dtype* gpu_data() const;
  const Dtype* gpu_diff() const;
  const Dtype* gpu_binary() const; // added term
  const Dtype* gpu_quantum() const; // added term
  const Dtype* gpu_acmean() const;  
  Dtype* mutable_gpu_data();
  Dtype* mutable_gpu_diff();
  Dtype* mutable_gpu_binary();  // added term
  Dtype* mutable_gpu_quantum();  // added term
  Dtype* mutable_gpu_acmean(); 
  const int* gpu_shape() const;

  void Update();
  void FromProto(const BlobProto& proto, bool reshape = true);
  // added @ 2016-4-13
  void BinaryFromProto(const BlobProto& proto, bool reshape = true);
  void TernaryFromProto(const BlobProto& proto, bool reshape = true);

  void ToProto(BlobProto* proto, bool write_diff = false) const;
  // added @ 2016-4-13
  void BinaryToProto(BlobProto* proto) const;
  void TernaryToProto(BlobProto* proto) const;

  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  Dtype asum_data() const;
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  Dtype asum_diff() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  Dtype sumsq_data() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  Dtype sumsq_diff() const;

  /// @brief Scale the blob data by a constant factor.
  void scale_data(Dtype scale_factor);
  /// @brief Scale the blob diff by a constant factor.
  void scale_diff(Dtype scale_factor);

  /// added functions
  // clip blob data
  void clip_data();
  // binarize data_ -> binary_
  void binarize_data(const Dtype bounds = 1);
  void ternarize_data(Phase phase = TRAIN);
  // quantize data_ -> quantum_
  void quantize_data(const Dtype left = -4, const Dtype right = 3);
  // acmean data_ -> acmean_
  void acmean_data();
  // scale binary: scale the binarized data for each feature map 
  void scale_binary();

  vector<int> ternary_stats();

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Blob& other);
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareDiff(const Blob& other);

  bool ShapeEquals(const BlobProto& other);

  void set_delta();
  void set_delta(Dtype delta);
  Dtype get_delta() const;

  void set_alpha(Dtype alpha);
  Dtype get_alpha() const;

 protected:
  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> diff_;
  shared_ptr<SyncedMemory> binary_;  // added term @ 2016-3-10
  shared_ptr<SyncedMemory> quantum_; // added term @ 2016-3-16
  shared_ptr<SyncedMemory> acmean_;  // added term @ 2016-3-18, absolute channel mean

  int acmean_len; // help property to store the length of acmean_

  Dtype delta_;
  Dtype alpha_;

  vector<int>* ternary_stats_;

  shared_ptr<SyncedMemory> shape_data_;
  vector<int> shape_;
  int count_;
  int capacity_;

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
