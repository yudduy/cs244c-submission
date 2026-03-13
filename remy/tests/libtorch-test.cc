#include <torch/torch.h>
#include <iostream>

int main() {
  std::cout << "LibTorch Test Program" << std::endl;

  // Create a simple 2x3 tensor filled with ones
  torch::Tensor tensor = torch::ones({2, 3});
  std::cout << "Created tensor:\n" << tensor << std::endl;

  // Check if CUDA is available (will be false for CPU-only build)
  std::cout << "CUDA available: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;

  std::cout << "LibTorch is properly installed and working!" << std::endl;

  return 0;
}
