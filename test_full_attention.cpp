#include "simulator.hpp"
#include <iostream>
#include <vector>

int main() {
  sjtu::GpuSimulator gpu_sim;
  sjtu::MatrixMemoryAllocator alloc;
  
  // Q: 2x3, K: 2x3, V: 2x3
  std::vector<float> Q_data = {1, 2, 3,
                                4, 5, 6};
  std::vector<float> K1_data = {1, 0, 0};
  std::vector<float> K2_data = {0, 1, 0};
  std::vector<float> V1_data = {1, 1, 1};
  std::vector<float> V2_data = {2, 2, 2};
  
  sjtu::Matrix* Q = new sjtu::Matrix(2, 3, Q_data, gpu_sim);
  sjtu::Matrix* K1 = new sjtu::Matrix(1, 3, K1_data, gpu_sim);
  sjtu::Matrix* K2 = new sjtu::Matrix(1, 3, K2_data, gpu_sim);
  sjtu::Matrix* V1 = new sjtu::Matrix(1, 3, V1_data, gpu_sim);
  sjtu::Matrix* V2 = new sjtu::Matrix(1, 3, V2_data, gpu_sim);
  
  // Concatenate
  sjtu::Matrix* K_all = alloc.Allocate("K_all");
  gpu_sim.Concat(K1, K2, K_all, 0, sjtu::kInGpuHbm);
  sjtu::Matrix* V_all = alloc.Allocate("V_all");
  gpu_sim.Concat(V1, V2, V_all, 0, sjtu::kInGpuHbm);
  gpu_sim.Run(false, &alloc);
  
  // Move to SRAM and compute
  gpu_sim.MoveMatrixToSharedMem(Q);
  gpu_sim.MoveMatrixToSharedMem(K_all);
  gpu_sim.Transpose(K_all, sjtu::kInSharedMemory);
  sjtu::Matrix* QKT = alloc.Allocate("QKT");
  gpu_sim.MatMul(Q, K_all, QKT);
  
  // Softmax
  sjtu::Matrix* softmax_result = nullptr;
  for (size_t r = 0; r < 2; ++r) {
    sjtu::Matrix* row = alloc.Allocate("row_" + std::to_string(r));
    gpu_sim.GetRow(QKT, r, row, sjtu::kInSharedMemory);
    sjtu::Matrix* exp_row = alloc.Allocate("exp_row_" + std::to_string(r));
    gpu_sim.MatExp(row, exp_row);
    sjtu::Matrix* sum_exp = alloc.Allocate("sum_exp_" + std::to_string(r));
    gpu_sim.Sum(exp_row, sum_exp);
    sjtu::Matrix* softmax_row = alloc.Allocate("softmax_row_" + std::to_string(r));
    gpu_sim.MatDiv(exp_row, sum_exp, softmax_row);
    
    if (r == 0) {
      softmax_result = alloc.Allocate("softmax_result");
      gpu_sim.Copy(softmax_row, softmax_result, sjtu::kInSharedMemory);
    } else {
      sjtu::Matrix* softmax_prev = softmax_result;
      softmax_result = alloc.Allocate("softmax_result_" + std::to_string(r));
      gpu_sim.Concat(softmax_prev, softmax_row, softmax_result, 0, sjtu::kInSharedMemory);
    }
  }
  
  // Transpose K_all back and compute final result
  gpu_sim.Transpose(K_all, sjtu::kInSharedMemory);
  gpu_sim.MoveMatrixToSharedMem(V_all);
  sjtu::Matrix* result = alloc.Allocate("result");
  gpu_sim.MatMul(softmax_result, V_all, result);
  
  gpu_sim.Run(false, &alloc);
  
  std::cerr << "Final result (should be ~[1.73, 1.73, 1.73; 1.73, 1.73, 1.73]):" << std::endl;
  result->Print();
  
  // Expected:
  // Row 0: 0.27 * [1,1,1] + 0.73 * [2,2,2] = [1.73, 1.73, 1.73]
  // Row 1: 0.27 * [1,1,1] + 0.73 * [2,2,2] = [1.73, 1.73, 1.73]
  
  return 0;
}
