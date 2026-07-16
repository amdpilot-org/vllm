#include <torch/extension.h>
#include "attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attention", &paged_attention, "Custom paged attention (ll4mi MFMA16)");
}
