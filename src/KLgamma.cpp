#include "RcppArmadillo.h"
#include "KLgamma.h"
// [[Rcpp::depends(RcppArmadillo)]]

double klgamma_sub(double a, double b, double c, double d){
  return - c * d / a - b * log(a) - lgamma(1/b) + (b-1)*(R::digamma(1/d) + log(c));
}

double kl2gamma(double a1, double b1, double a2, double b2){
  return klgamma_sub(a2,b2,a2,b2) - klgamma_sub(a1,b1,a2,b2);
}

arma::mat mat_digamma(arma::mat & a){
  int K = a.n_rows;
  int L = a.n_cols;
  arma::mat out(K,L);
  for(int k=0;k<K;k++){
    for(int l=0;l<L;l++){
      out(k,l) = R::digamma(a(k,l));
    }
  }
  return out;
}

arma::vec vec_digamma(const arma::vec & a){
  int K = a.n_rows;
  arma::vec out(K);
  for(int k=0;k<K;k++){
    out(k) = R::digamma(a(k));
  }
  return out;
}

double lowerbound_logML_pois(const arma::vec & y,
                             const arma::mat & alpha_z,
                             const arma::mat & beta_z,
                             const arma::mat & alpha_w,
                             const arma::mat & beta_w,
                             const arma::mat & Z,
                             const arma::mat & W,
                             const arma::mat & logZ,
                             const arma::mat & logW,
                             const double & a,
                             const double & b){
  double lp = 0;
  lp += sum(sum((a-1)*logZ - b*Z,0) +a*log(beta_z) - Z.n_rows*std::lgamma(a));
  lp -= accu((alpha_z-1)%logZ - Z.each_row()%beta_z + alpha_z.each_row()%log(beta_z) - lgamma(alpha_z));
  lp += sum(sum((a-1)*logW - b*W,0) + a*log(beta_w) - W.n_rows*W.n_elem*std::lgamma(a));
  lp -= accu((alpha_w-1)%logW - W.each_row()%beta_w + alpha_w.each_row()%log(beta_w) - lgamma(alpha_w));
  return lp;
}

void up_log_gamma(arma::vec & logv, const arma::vec & a, const arma::vec & logb){
  int K = logv.n_rows;
  for(int k=0;k<K;k++){
    logv(k) = R::digamma(a(k)) - logb(k);
  }
}

double kld(const arma::mat & alpha,
           const arma::mat & beta,
           const double & a,
           const double & b){
  double lp = 0;
  for(int i=0; i<alpha.n_rows; i++){
    for(int l=0; l<alpha.n_cols; l++){
      lp += kl2gamma(a, b, alpha(i,l), beta(i,l));
    }
  }
  return lp;
}

double outerprod(arma::vec v){
  int K=v.n_rows;
  double out = 0;
  for(int i=0; i<K-1; i++){
    for(int j=i+1; j<K; j++){
      out += v(i)*v(j);
    }
  }
  return out;
}
