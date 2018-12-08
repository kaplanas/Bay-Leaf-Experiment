data {
  int<lower=1> N; // number of observations
  int<lower=0,upper=1> prefer_bay_leaf[N]; // outcome of observation n
}
parameters {
  real beta; // overall intercept; this is the parameter of interest
}
model {
  // priors
  beta ~ normal(0, 2);
  // likelihood
  prefer_bay_leaf ~ bernoulli_logit(beta);
}
