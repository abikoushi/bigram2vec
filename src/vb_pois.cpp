#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;
#include "KLgamma.h"
#include "rand.h"
#include <progress.hpp>
#include <progress_bar.hpp>
// [[Rcpp::depends(RcppProgress)]]

//shape parameters
double up_A(arma::mat & alpha,
            const arma::mat & logV,
            const arma::vec & y,
            const arma::umat & X,
            const double & a,
            const int & L,
            const int & Dim){
  //initialize by hyper parameter
  alpha.fill(a);
  double lp = 0;
  //inclement sufficient statistics
  for(int n=0; n<y.n_rows; n++){
    arma::rowvec num = arma::zeros<arma::rowvec>(L);
    for(int j = 0; j < X.n_cols; j++){
      num += logV.row(X(n,j));
    }
    rowvec r = exp(num);
    double R = sum(r);
    r = y(n)*(r/R);
    for(int j = 0; j < X.n_cols; j++){
      alpha.row(X(n,j)) += r;
    }
    lp +=  y(n)*log(R);
  }
  return lp;
}

double up_B(const arma::mat & alpha,
            arma::mat & beta,
            arma::mat & V,
            arma::mat & logV,
            const double & b,
            const int & L,
            const int & Dim){
  double lp = 0;
  for(int l=0; l<L; l++){
    double sumV = sum(V.col(l));
    beta.col(l) = sumV*arma::ones(Dim) - V.col(l) + b;
    arma::vec Vl = alpha.col(l)/beta.col(l);
    V.col(l) = Vl;
    arma::vec logv = logV.col(l);
    up_log_gamma(logv, alpha.col(l), log(beta.col(l)));
    logV.col(l) = logv;
    lp -= outerprod(V.col(l));
  }
  return lp;
}

double up_theta(arma::mat & alpha,
                arma::mat & beta,
                arma::mat & V,
                arma::mat & logV,
                const arma::vec & y,
                const arma::umat & X,
                const int & L,
                const int & Dim,
                const double & a,
                const double & b){
  double lp_a = 0;
  double lp_b = 0;
  for(int k=0; k < Dim; k++){
    lp_a += up_A(alpha, logV, y, X, a, L, Dim);
    lp_b += up_B(alpha, beta, V, logV, b, L, Dim);
  }
  return lp_a+lp_b;
}


// [[Rcpp::export]]
List doVB_pois(const arma::vec & y,
               const arma::umat & X,
               const int & Dim,
               const int & L,
               const int & iter,
               const double & a,
               const double & b,
               const bool & display_progress){
  arma::mat V = randg<arma::mat>(Dim,L);
  arma::mat logV = log(V);
  arma::mat alpha = arma::ones<arma::mat>(Dim, L);
  arma::mat beta = arma::ones<arma::mat>(Dim, L);
  arma::vec lp = arma::zeros<arma::vec>(iter);
  Progress pb(iter, display_progress);
  for(int i=0; i<iter; i++){
    double lp0 = up_theta(alpha, beta, V, logV, y, X, L, Dim, a, b);
    lp(i) = lp0 + kld(alpha, beta, a, b);
    pb.increment();
  }
  lp -= sum(lgamma(y+1));
  return List::create(Named("shape")=alpha,
                      Named("rate")=beta,
                      Named("logprob")=lp);
}
