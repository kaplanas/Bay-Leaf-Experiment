data {
  int<lower=1> N; // number of observations
  int<lower=0> I; // number of batches; if 0, we won't model batch
  int<lower=0> J; // number of subjects; if 0, we won't model subject
  int<lower=0> K; // number of pairs; if 0, we won't model pair
  int<lower=(I==0?1:1),upper=(I==0?1:I)> batch[N]; // batch of observation n; always 1 if not modeled
  int<lower=(J==0?1:1),upper=(J==0?1:J)> subject[N]; // subject of observation n; always 1 if not modeled
  int<lower=(K==0?1:1),upper=(K==0?1:K)> pair[N]; // pair of observation n; always 1 if not modeled
  int<lower=0,upper=1> prefer_bay_leaf[N]; // outcome of observation n
}
parameters {
  real beta; // overall intercept; this is the parameter of interest
  vector[I] gamma; // intercept for batch i
  real<lower=0> sigma_gamma[I == 0 ? 0 : 1]; // standard deviation of by-batch intercepts, if modeled
  vector[J] delta; // intercept for subject j
  real<lower=0> sigma_delta[J == 0 ? 0 : 1]; // standard deviation of by-subject intercepts, if modeled
  vector[K] epsilon; // intercept for pair k
  real<lower=0> sigma_epsilon[K == 0 ? 0 : 1]; // standard deviation of by-pair intercepts, if modeled
}
model {
  // priors
  beta ~ normal(0, 2);
  sigma_gamma ~ cauchy(0, 1);
  gamma ~ normal(0, 1);
  sigma_delta ~ cauchy(0, 1);
  delta ~ normal(0, 1);
  sigma_epsilon ~ cauchy(0, 1);
  epsilon ~ normal(0, 1);
  // likelihood
  {
    vector[N] mu;
    mu = rep_vector(beta, N);
    if(I > 0) {
      mu += gamma[batch] * sigma_gamma[1];
    }
    if(J > 0) {
      mu += delta[subject] * sigma_delta[1];
    }
    if(K > 0) {
      mu += epsilon[pair] * sigma_epsilon[1];
    }
    prefer_bay_leaf ~ bernoulli_logit(mu);
  }
}
