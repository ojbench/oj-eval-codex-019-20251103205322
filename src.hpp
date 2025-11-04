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
    K_all = matrix_memory_allocator.Allocate("K_temp_" + std::to_string(j));
    gpu_sim.Concat(K_prev, keys[j], K_all, 0, kInGpuHbm);
  }
  
  // Concatenate all values in HBM to create V_all (32 x 512)
  Matrix* V_all = matrix_memory_allocator.Allocate("V_all");
  gpu_sim.Concat(values[0], values[1], V_all, 0, kInGpuHbm);
  for (size_t j = 2; j < values.size(); ++j) {
    Matrix* V_prev = V_all;
    V_all = matrix_memory_allocator.Allocate("V_temp_" + std::to_string(j));
    gpu_sim.Concat(V_prev, values[j], V_all, 0, kInGpuHbm);
  }
  
  // Run the concatenation operations
  gpu_sim.Run(false, &matrix_memory_allocator);
  
  // Now process each query
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    
    // Move query and K_all to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(K_all);
    
    // Transpose K_all: (32 x 512) -> (512 x 32)
    gpu_sim.Transpose(K_all, kInSharedMemory);
    
    // Compute Q * K^T: (i+1 x 512) * (512 x 32) = (i+1 x 32)
    Matrix* QKT = matrix_memory_allocator.Allocate("QKT_" + std::to_string(i));
    gpu_sim.MatMul(current_query, K_all, QKT);
    
    // Apply Softmax row-wise
    size_t num_rows = i + 1;
    Matrix* softmax_result = nullptr;
    
    // Process each row
    for (size_t r = 0; r < num_rows; ++r) {
      // Get row r
      Matrix* row = matrix_memory_allocator.Allocate("row_" + std::to_string(i) + "_" + std::to_string(r));
      gpu_sim.GetRow(QKT, r, row, kInSharedMemory);
      
      // Compute exp(row)
      Matrix* exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(i) + "_" + std::to_string(r));
      gpu_sim.MatExp(row, exp_row);
      
      // Sum all elements in exp_row
      Matrix* sum_exp = matrix_memory_allocator.Allocate("sum_exp_" + std::to_string(i) + "_" + std::to_string(r));
      gpu_sim.Sum(exp_row, sum_exp);
      
      // Divide exp_row by sum_exp
      Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row_" + std::to_string(i) + "_" + std::to_string(r));
      gpu_sim.MatDiv(exp_row, sum_exp, softmax_row);
      
      // Concatenate to softmax_result
      if (r == 0) {
        softmax_result = matrix_memory_allocator.Allocate("softmax_" + std::to_string(i));
        gpu_sim.Copy(softmax_row, softmax_result, kInSharedMemory);
      } else {
        Matrix* softmax_prev = softmax_result;
        softmax_result = matrix_memory_allocator.Allocate("softmax_temp_" + std::to_string(i) + "_" + std::to_string(r));
        gpu_sim.Concat(softmax_prev, softmax_row, softmax_result, 0, kInSharedMemory);
      }
    }
    
    // Transpose K_all back: (512 x 32) -> (32 x 512)
    gpu_sim.Transpose(K_all, kInSharedMemory);
    
    // Move V_all to SRAM
    gpu_sim.MoveMatrixToSharedMem(V_all);
    
    // Compute softmax_result * V_all: (i+1 x 32) * (32 x 512) = (i+1 x 512)
    Matrix* result = matrix_memory_allocator.Allocate("result_" + std::to_string(i));
    gpu_sim.MatMul(softmax_result, V_all, result);
    
    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);
    
    // Run all operations for this iteration
    gpu_sim.Run(false, &matrix_memory_allocator);
    
    // Commit the answer
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
