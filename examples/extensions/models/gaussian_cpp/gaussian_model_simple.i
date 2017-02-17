%module gaussian_model_simple
%{
  #define SWIG_FILE_WITH_INIT
  
  #include <iostream>
  #include <boost/random.hpp>
  #include <boost/random/normal_distribution.hpp>
  
  extern void gaussian_model(double* result, unsigned int k, double mu, double sigma, boost::mt19937 rng);
%}

%include "numpy.i"

%init %{
  import_array();
%}

%inline %{
  boost::mt19937* get_rng(int seed) {
    boost::mt19937* rng = new boost::mt19937(seed);
    return rng;
  }
%}

%apply (double* ARGOUT_ARRAY1, int DIM1 ) {(double* result, unsigned int k)};

extern void gaussian_model(double* result, unsigned int k, double mu, double sigma, boost::mt19937 rng);

