data {
  int<lower=1> N; // number of observations
  int<lower=1> I; // number of batches
  int<lower=1> J; // number of subjects
  int<lower=1> K; // number of pairs
  int<lower=1,upper=I> batch[N]; // batch of observation n
  int<lower=1,upper=J> subject[N]; // subject of observation n
  int<lower=1,upper=K> pair[N]; // pair of observation n
  int<lower=0,upper=1> prefer_bay_leaf[N]; // outcome of observation n
}
parameters {
  real beta; // overall intercept; this is the parameter of interest
  vector[I] gamma; // intercept for batch i
  real<lower=0> sigma_gamma; // standard deviation of by-batch intercepts
  vector[J] delta; // intercept for subject j
  real<lower=0> sigma_delta; // standard deviation of by-subject intercepts
  vector[K] epsilon; // intercept for pair k
  real<lower=0> sigma_epsilon; // standard deviation of by-pair intercepts
}
model {
  // priors
  beta ~ normal(0, 1);
  sigma_gamma ~ cauchy(0, 1);
  gamma ~ normal(0, 1);
  sigma_delta ~ cauchy(0, 1);
  delta ~ normal(0, 1);
  sigma_epsilon ~ cauchy(0, 1);
  epsilon ~ normal(0, 1);
  // likelihood
  prefer_bay_leaf ~ bernoulli_logit(beta
                                    + (gamma[batch] * sigma_gamma)
                                    + (delta[subject] * sigma_delta)
                                    + (epsilon[pair] * sigma_epsilon));
}
