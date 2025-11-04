#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  
  // Concatenate all keys in HBM to create K_all (32 x 512)
  Matrix* K_all = matrix_memory_allocator.Allocate("K_all");
  gpu_sim.Concat(keys[0], keys[1], K_all, 0, kInGpuHbm);
  for (size_t j = 2; j < keys.size(); ++j) {
    Matrix* K_prev = K_all;
    K_all = matrix_memory_allocator.Allocate("K_all_" + std::to_string(j));
    gpu_sim.Concat(K_prev, keys[j], K_all, 0, kInGpuHbm);
  }
  
  // Concatenate all values in HBM to create V_all (32 x 512)
  Matrix* V_all = matrix_memory_allocator.Allocate("V_all");
  gpu_sim.Concat(values[0], values[1], V_all, 0, kInGpuHbm);
  for (size_t j = 2; j < values.size(); ++j) {
    Matrix* V_prev = V_all;
    V_all = matrix_memory_allocator.Allocate("V_all_" + std::to_string(j));
    gpu_sim.Concat(V_prev, values[j], V_all, 0, kInGpuHbm);
  }
  
  // Run concatenation operations
  gpu_sim.Run(false, &matrix_memory_allocator);
  
  // Process each query
  for (size_t i = 0; i < keys.size(); ++i) {
    auto Q = rater.GetNextQuery();
    size_t num_rows = Q->GetRowNum();
    
    // Move matrices to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(Q);
    gpu_sim.MoveMatrixToSharedMem(K_all);
    gpu_sim.MoveMatrixToSharedMem(V_all);
    
    // Transpose K_all: (32 x 512) -> (512 x 32)
    gpu_sim.Transpose(K_all, kInSharedMemory);
    
    // Compute Q * K^T
    Matrix* QKT = matrix_memory_allocator.Allocate("QKT_" + std::to_string(i));
    gpu_sim.MatMul(Q, K_all, QKT);
    
    // Apply softmax row-wise
    Matrix* softmax_result = nullptr;
    for (size_t r = 0; r < num_rows; ++r) {
      Matrix* row = matrix_memory_allocator.Allocate("row_" + std::to_string(i) + "_" + std::to_string(r));
      gpu_sim.GetRow(QKT, r, row, kInSharedMemory);
      
      Matrix* exp_row = matrix_memory_allocator.Allocate("exp_" + std::to_string(i) + "_" + std::to_string(r));
      gpu_sim.MatExp(row, exp_row);
      
      Matrix* sum = matrix_memory_allocator.Allocate("sum_" + std::to_string(i) + "_" + std::to_string(r));
      gpu_sim.Sum(exp_row, sum);
      
      Matrix* softmax_row = matrix_memory_allocator.Allocate("sfrow_" + std::to_string(i) + "_" + std::to_string(r));
      gpu_sim.MatDiv(exp_row, sum, softmax_row);
      
      if (r == 0) {
        softmax_result = matrix_memory_allocator.Allocate("softmax_" + std::to_string(i));
        gpu_sim.Copy(softmax_row, softmax_result, kInSharedMemory);
      } else {
        Matrix* prev = softmax_result;
        softmax_result = matrix_memory_allocator.Allocate("softmax_" + std::to_string(i) + "_" + std::to_string(r));
        gpu_sim.Concat(prev, softmax_row, softmax_result, 0, kInSharedMemory);
      }
    }
    
    // Transpose K_all back
    gpu_sim.Transpose(K_all, kInSharedMemory);
    
    // Compute final result
    Matrix* result = matrix_memory_allocator.Allocate("result_" + std::to_string(i));
    gpu_sim.MatMul(softmax_result, V_all, result);
    
    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);
    
    // Execute all operations
    gpu_sim.Run(false, &matrix_memory_allocator);
    
    // Commit answer
    rater.CommitAnswer(*result);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
