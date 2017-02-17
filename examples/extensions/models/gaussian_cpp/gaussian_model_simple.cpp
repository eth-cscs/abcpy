#include <iostream>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

using namespace std;

// Simulation function of the gaussian model
void gaussian_model(double* result, unsigned int k, double mu, double sigma, boost::mt19937 rng) {
  boost::normal_distribution<> nd(mu, sigma);
  boost::variate_generator<boost::mt19937, boost::normal_distribution<> > sampler(rng, nd);
  
  for (int i=0; i<k; ++i) {
    result[i] = sampler();
  }
}



// main function to run the simulation of the Gaussian model
int main() {
  int k = 10;
  boost::mt19937 rng;
  double samples[k];
  gaussian_model(samples, 0.0, 1.0, k, rng);
  
  for (int i=0; i<k; ++i) {
    std::cout << samples[i] << " ";
    std::cout << std::endl;
  }
  
  return 0;
}
