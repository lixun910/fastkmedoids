\name{fastkmedoids-package}
\alias{fastkmedoids-package}
\alias{fastkmedoids}
\docType{package}
\title{
  Faster K-Medoids Clustering Algorithms: FastPAM, FastCLARA, FastCLARANS
}
\description{
  This package provides R wrappers of C++ implementation of Faster K-Medoids
  clustering algorithms (FastPAM, FastCLARA and FastCLARANS) 
  proposed in Erich Schubert and Peter J. Rousseeuw 2019.
}
\details{
  The C++ Faster K-Medoids clustering algorithms (FastPAM, FastCLARA and FastCLARANS) 
  are ported from ELKI project (see http://elki-project.github.io/). To generate identical
  results, the random number generator, specifically xorshift+ generators, is also ported.
  The results between this fastkmedoids R package will be the same with ELKI if using
  same seed for random number generator. 
  
  Besides FastPAM, FastCLARA and FastCLARANS, this R wrapper also provide However,
  in the C++ code, the classic algorithms, including PAM, CLARA and CLARANS, are also
  implemented. If interested in writing wrappers for these algorithms, please use the 
  github repository: https://github.com/lixun910/fastkmedoids
  
  All three algorithms take the distance matrix (lower triangular part, column wise storage)
  as input, which can be computed using dist() function in R (see the examples below). If
  using a pre-computed distance matrix, please transform it (lower triangular part, 
  column wise storage) to a 1-dimensional array.
  
  All three algorithms takes the same parameters as in ELKI. If the explanation of the 
  input paramters is not clear, please refer to ELKI :
  
  FastPAM: https://elki-project.github.io/releases/current/javadoc/de/lmu/ifi/dbs/elki/algorithm/clustering/kmeans/KMedoidsFastPAM.html
  FastCLARA: https://elki-project.github.io/releases/current/javadoc/de/lmu/ifi/dbs/elki/algorithm/clustering/kmeans/FastCLARA.html
  FastCLARANS: https://elki-project.github.io/releases/current/javadoc/de/lmu/ifi/dbs/elki/algorithm/clustering/kmeans/FastCLARANS.html
  
  The C++ code is a part of GeoDa (https://github.com/geodacenter/geoda) and libgeoda. 
  If you are interested in a GUI version of this C++ implementation. You can download
  and use the free and cross-platform GeoDa software from https://geodacenter.github.io.
  
}
\author{
Xun Li
Maintainer: Xun Li <lixun910@gmail.com>
}
\references{
  Erich Schubert, Peter J. Rousseeuw  "Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms" 2019 
  <doi:10.1007/978-3-030-32047-8_16>
}
\keyword{ fastkmedoids cluster kmedoid pam clara clarans }
\seealso{
  https://arxiv.org/abs/1810.05691
}
\examples{
  # We use the demo data sets "USArrests"
  data("USArrests")
  df <- scale(USArrests)
  dv <- as.vector(dist(df)) # compute distance matrix (lower triangular part)
  n <- nrow(df)
  
  # PAM
  clusters <- pam(dv, n, k=3)
  clusters
  
  # FastPAM (use "LAB" initializer by default)
  clusters1 <- fastpam(dv, n, k=3)
  clusters1
  
  # FastPAM, specify "BUILD" as initializer
  clusters2 <- fastpam(dv, n, k=3, initializer="BUILD")
  clusters2
  
  # FastCLARA
  clusters3 <- fastclara(dv, n, k=3, numsamples = 5, sampling=0.25)
  clusters3
  
  # FastCLARANS
  clusters4 <- fastclarans(dv, n, k=3, numlocal=2, maxneighbor=0.025)
  clusters4
}
