// Author: Xun Li <lixun910@gmail.com>
// May 27, 2020
//
// Code ported from elki project: http://elki-project.github.io/
// Copyright follows elki project: http://elki-project.github.io/
// AGPLv3: https://github.com/elki-project/elki/blob/master/LICENSE.md
//
// 5-27-2020
// Xoroshiro128Random random number generator
// PAM, CLARA, CLARANS
// Initializer: BUILD and LAB
// FastPAM, FastCLARA, FastCLARANS

// [[Rcpp::plugins("cpp11")]]

#include <Rcpp.h>
using namespace Rcpp;

#include "pam.h"

//' PAM (Partitioning Around Medoids)
//' 
//' @description  The original Partitioning Around Medoids (PAM) algorithm or k-medoids
//' clustering, as proposed by Kaufman and Rousseeuw; a largely equivalent method
//' was also proposed by Whitaker in the operations research domain, and is well
//' known by the name "fast interchange" there.
//' (Schubert and Rousseeuw, 2019)
//' 
//' @references L. Kaufman, P. J. Rousseeuw
//' "Clustering by means of Medoids"
//' Information Systems and Operational Research 21(2)
//' 
//' @param rdist The distance matrix (lower triangular matrix, column wise storage)
//' @param n The number of observations
//' @param k The number of clusters to produce
//' @param maxiter The maximum number of iterations (default: 0)
//' @return KMedoids S4 class
//' @export
// [[Rcpp::export]]
S4 pam(NumericVector rdist, int n, int k, int maxiter = 0) {
  //CharacterVector x = CharacterVector::create( "foo", "bar" )  
  //NumericVector y   = NumericVector::create( dist.length() ) ;
  //List z            = List::create( x, y ) ;
  std::vector<double> dist = as<std::vector<double> >(rdist);
  
  RDistMatrix dm(n, dist);
  
  BUILD init(&dm);
  
  PAM pam(n, &dm, &init, k, maxiter);
  
  double cost = pam.run();
  
  std::vector<int> medoids = pam.getMedoids();
  std::vector<int> results = pam.getResults();
  
  S4 r("KmedoidsResult");
  r.slot("cost") = cost;
  r.slot("medoids") = medoids;
  r.slot("assignment") = results;
  
  return r ;
}

//' FastPAM
//' 
//' @description FastPAM: An improved version of PAM, that is usually O(k) times faster.
//' Because of the speed benefits, we also suggest to use a linear-time
//' initialization, such as the k-means++ initialization or the proposed
//' LAB (linear approximative BUILD, the third component of FastPAM)
//' initialization, and try multiple times if the runtime permits.
//' (Schubert and Rousseeuw, 2019)
//' 
//' @references Erich Schubert, Peter J. Rousseeuw 
//' "Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms"
//' 2019 https://arxiv.org/abs/1810.05691
//' @param rdist The distance matrix (lower triangular matrix, column wise storage)
//' @param n The number of observations
//' @param k The number of clusters to produce.
//' @param maxiter The maximum number of iterations (default: 0)
//' @param initializer Initializer: either "BUILD" (used in classic PAM) or "LAB" (linear approximative BUILD)
//' Because of the speed benefits, "LAB" is suggested, and one can try multiple times if the runtime permits.
//' @param fasttol Tolerance for fast swapping behavior (may perform worse swaps). 
//' Default: 1.0, which means to perform any additional swap that gives an improvement.
//' When set to 0, it will only execute an additional swap if it appears to be independent
//' (i.e., the improvements resulting from the swap have not decreased when the first swap was executed).
//' @param seed Seed for random number generator. Default: 123456789
//' @return KMedoids S4 class
//' @export
// [[Rcpp::export]]
S4 fastpam(NumericVector rdist, int n, int k, int maxiter=0, std::string initializer="LAB", 
           double fasttol = 1.0, int seed = 123456789) {
  std::vector<double> dist = as<std::vector<double> >(rdist);
  
  RDistMatrix dm(n, dist);
  
  PAMInitializer* init;
  if (initializer.compare("BUILD")) {
    init = new BUILD(&dm);
  } else {
    init = new LAB(&dm, seed);
  }
  
  FastPAM pam(n, &dm, init, k, maxiter, fasttol);
  
  double cost = pam.run();
  
  std::vector<int> medoids = pam.getMedoids();
  std::vector<int> results = pam.getResults();
  
  delete init;
  
  S4 r("KmedoidsResult");
  r.slot("cost") = cost;
  r.slot("medoids") = medoids;
  r.slot("assignment") = results;
  
  return r ;
}

//' FastCLARA
//' 
//' @description Clustering Large Applications (CLARA) with the
//'  improvements, to increase scalability in the number of clusters. This variant
//'  will also default to twice the sample size, to improve quality. 
//'  (Schubert and Rousseeuw, 2019)
//'  
//' @references Erich Schubert, Peter J. Rousseeuw 
//' "Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms"
//' 2019 https://arxiv.org/abs/1810.05691
//' 
//' @param rdist The distance matrix (lower triangular matrix, column wise storage)
//' @param n The number of observations
//' @param k The number of clusters to produce
//' @param maxiter The maximum number of iterations (default: 0)
//' @param initializer Initializer: either "BUILD" (used in classic PAM) or "LAB" (linear approximative BUILD)
//' @param fasttol Tolerance for fast swapping behavior (may perform worse swaps). 
//' Default: 1.0, which means to perform any additional swap that gives an improvement.
//' When set to 0, it will only execute an additional swap if it appears to be independent
//' (i.e., the improvements resulting from the swap have not decreased when the first swap was executed).
//' @param numsamples Number of samples to draw (i.e. iterations). Default: 5
//' @param sampling Sampling rate. Default value: 80 + 4*k. (see Schubert and Rousseeuw, 2019)
//'   If less than 1, it is considered to be a relative value. e.g. N*0.10
//' @param independent NOT Keep the previous medoids in the next sample. Default: FALSE
//' @param seed Seed for random number generator. Default: 123456789
//' @return KMedoids S4 class
//' @export
// [[Rcpp::export]]
S4 fastclara(NumericVector rdist, int n, int k, int maxiter=0, std::string initializer="LAB", 
             double fasttol = 1.0, int numsamples = 5, double sampling = 0.25, 
             bool independent = false, int seed = 123456789) {
  std::vector<double> dist = as<std::vector<double> >(rdist);
  
  RDistMatrix dm(n, dist);
  
  PAMInitializer* init;
  if (initializer.compare("BUILD")) {
    init = new BUILD(&dm);
  } else {
    init = new LAB(&dm);
  }
  
  FastCLARA clara(n, &dm, init, k, maxiter, fasttol, numsamples, sampling, independent, seed);
  
  double cost = clara.run();
  
  std::vector<int> medoids = clara.getMedoids();
  std::vector<int> results = clara.getResults();
  
  delete init;
  
  S4 r("KmedoidsResult");
  r.slot("cost") = cost;
  r.slot("medoids") = medoids;
  r.slot("assignment") = results;
  
  return r ;
}

//' FastCLARANS
//' 
//' @description A faster variation of CLARANS, that can explore O(k) as many swaps at a
//'  similar cost by considering all medoids for each candidate non-medoid. Since
//'  this means sampling fewer non-medoids, we suggest to increase the subsampling
//'  rate slightly to get higher quality than CLARANS, at better runtime. 
//'  (Schubert and Rousseeuw, 2019)
//'  
//' @references Erich Schubert, Peter J. Rousseeuw 
//' "Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms"
//' 2019 https://arxiv.org/abs/1810.05691
//' 
//' @param rdist The distance matrix (lower triangular matrix, column wise storage)
//' @param n The number of observations
//' @param k The number of clusters to produce.
//' @param numlocal  Number of samples to draw (i.e. restarts).
//'   Default: 2
//' @param maxneighbor Sampling rate. If less than 1, it is considered to be a relative value.
//'   Default: 2 * 0.0125, larger sampling rate than CLARANS (see Schubert and Rousseeuw, 2019)
//' @param seed Seed for random number generator. Default: 123456789
//' @return KMedoids S4 class
//' @export
// [[Rcpp::export]]
S4 fastclarans(NumericVector rdist, int n, int k, 
               int numlocal=2, double maxneighbor=0.025, int seed = 123456789) {
  std::vector<double> dist = as<std::vector<double> >(rdist);
  
  RDistMatrix dm(n, dist);
  
  FastCLARANS clarans(n, &dm, k, numlocal, maxneighbor, seed);
  
  double cost = clarans.run();
  
  std::vector<int> medoids = clarans.getMedoids();
  std::vector<int> results = clarans.getResults();
  
  S4 r("KmedoidsResult");
  r.slot("cost") = cost;
  r.slot("medoids") = medoids;
  r.slot("assignment") = results;
  
  return r ;
}