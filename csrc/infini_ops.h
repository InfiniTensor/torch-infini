#ifndef TORCH_INFINI_INFINI_OPS_H_
#define TORCH_INFINI_INFINI_OPS_H_

#include <ATen/ATen.h>
#include <infini/ops.h>
#include <infini/rt.h>

namespace torch_infini::infini_ops {

struct ExecutionContext {
  infini::ops::Handle handle;
  infini::ops::Config config;
};

infini::rt::DataType to_data_type(c10::ScalarType scalar_type);

infini::rt::Device to_device(const c10::Device& device);

infini::rt::TensorView::Shape to_shape(c10::IntArrayRef sizes);

infini::rt::TensorView::Strides to_strides(c10::IntArrayRef strides);

infini::rt::TensorView to_tensor_view(const at::Tensor& tensor);

ExecutionContext make_execution_context(infini::rt::runtime::Stream stream);

} // namespace torch_infini::infini_ops

#endif // TORCH_INFINI_INFINI_OPS_H_
