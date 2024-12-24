double klgamma_sub(double a, double b, double c, double d);

double kl2gamma(double a1, double b1, double a2, double b2);

arma::mat mat_digamma(arma::mat & a);

arma::vec vec_digamma(const arma::vec & a);

void up_log_gamma(arma::vec & logv, const arma::vec & a, const arma::vec & logb);

double kld(const arma::mat & alpha,
            const arma::mat & beta,
            const double & a,
            const double & b);

double outerprod(arma::vec v);
