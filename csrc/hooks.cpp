#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <mutex>

#include "torch_infini.h"

namespace torch_infini {

namespace {

class Hooks final : public at::PrivateUse1HooksInterface {
 public:
  c10::Allocator* getPinnedMemoryAllocator() const override {
    return get_host_allocator();
  }

  bool isPinnedPtr(const void* data) const override {
    return torch_infini::is_pinned_ptr(data);
  }
};

std::once_flag g_hooks_registration_once;

} // namespace

void register_privateuse1_hooks() {
  std::call_once(g_hooks_registration_once, [] {
    static auto* hooks = new Hooks();
    at::RegisterPrivateUse1HooksInterface(hooks);
  });
}

} // namespace torch_infini
