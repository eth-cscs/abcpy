simple_gaussian <- function(mu, sigma, k = 1){
	output <- rnorm(k, mu, sigma)
	return(output)
}