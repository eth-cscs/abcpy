simple_gaussian <- function(mu, sigma, k = 1, seed = seed) {
  set.seed(seed)
  output <- rnorm(k, mu, sigma)
  return(output)
}