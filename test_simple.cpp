#include "simulator.hpp"
#include <iostream>

int main() {
  sjtu::GpuSimulator gpu_sim;
  sjtu::MatrixMemoryAllocator alloc;
  
  // Create two simple 1x2 matrices in HBM
  std::vector<float> data1 = {1.0, 2.0};
  std::vector<float> data2 = {3.0, 4.0};
  
  sjtu::Matrix* m1 = new sjtu::Matrix(1, 2, data1, gpu_sim);
  sjtu::Matrix* m2 = new sjtu::Matrix(1, 2, data2, gpu_sim);
  
  std::cerr << "m1 position: " << m1->GetPosition() << std::endl;
  std::cerr << "m2 position: " << m2->GetPosition() << std::endl;
  
  // Try to concatenate
  sjtu::Matrix* result = alloc.Allocate("result");
  gpu_sim.Concat(m1, m2, result, 0, sjtu::kInGpuHbm);
  
  gpu_sim.Run(false, &alloc);
  
  std::cerr << "Result shape: " << result->GetRowNum() << "x" << result->GetColumnNum() << std::endl;
  result->Print();
  
  return 0;
}
