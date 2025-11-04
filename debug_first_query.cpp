#include "simulator.hpp"
#include "src.hpp"
#include <fstream>
#include <sstream>
#include <vector>

class DataLoader {
public:
  bool loadDataFromFile(const std::string &filename, std::vector<float> &data) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    std::string line;
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      float value;
      while (ss >> value) data.push_back(value);
    }
    file.close();
    return true;
  }
};

int main() {
  sjtu::GpuSimulator gpu_sim;
  sjtu::MatrixMemoryAllocator alloc;
  DataLoader loader;
  std::vector<float> key_data, value_data, query_data, ans_data;
  
  loader.loadDataFromFile("./data/keys.txt", key_data);
  loader.loadDataFromFile("./data/values.txt", value_data);
  loader.loadDataFromFile("./data/queries.txt", query_data);
  loader.loadDataFromFile("./data/ans.txt", ans_data);
  
  // Just process first query (1x512)
  sjtu::Matrix* Q = new sjtu::Matrix(1, 512, 
    std::vector<float>(query_data.begin(), query_data.begin() + 512), gpu_sim);
  alloc.Bind(Q, "Q");
  
  // Load all keys and values
  std::vector<sjtu::Matrix*> keys, values;
  for (int i = 0; i < 32; ++i) {
    keys.push_back(new sjtu::Matrix(1, 512,
      std::vector<float>(key_data.begin() + i*512, key_data.begin() + (i+1)*512), gpu_sim));
    values.push_back(new sjtu::Matrix(1, 512,
      std::vector<float>(value_data.begin() + i*512, value_data.begin() + (i+1)*512), gpu_sim));
  }
  
  // Expected answer
  sjtu::Matrix* expected = new sjtu::Matrix(1, 512,
    std::vector<float>(ans_data.begin(), ans_data.begin() + 512), gpu_sim);
  
  // Compute attention manually
  // ... (implement attention here)
  
  std::cerr << "Expected answer (first 10 values):" << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::cerr << expected->data_[i] << " ";
  }
  std::cerr << std::endl;
  
  return 0;
}
