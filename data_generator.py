import torch

class data_generator:



    def generate_nearly_unif(self, alphabet_size, beta, sample_size):
        p_vector = self._generate_power_law_p(alphabet_size, beta)
        return(
            self._generate_multinomial_data(p_vector, sample_size)
        )
 
    def generate_multinomial_data(self, p_vector, sample_size):
        return(
            torch.multinomial(
                p_vector,
                sample_size,
                replacement=True
            )
        )
    def _generate_power_law_p(self, alphabet_size, beta):
        p = torch.arange(1,alphabet_size+1).pow(-beta)
        p = p.divide(p.sum())
        return(p)
    
    def generate_power_law_p_private(self, alphabet_size, beta, privacy_level):
        p = torch.arange(1,alphabet_size+1).pow(-beta)
        p = p.divide(p.sum())
        exp_alpha = torch.tensor(privacy_level).exp()
        denumerator = exp_alpha.add(alphabet_size).sub(1)
        p_private = (p.mul(exp_alpha) + (1-p)).div(denumerator)
        
        return(p, p_private)
    
    def generate_copula_gaussian_data(self, sample_size, copula_mean, cov):
        cdf_calculator = torch.distributions.normal.Normal(loc = 0.0, scale = 1.0)
        generator_X = torch.distributions.multivariate_normal.MultivariateNormal(
            loc = copula_mean,
            covariance_matrix = cov
            )
        data_x = cdf_calculator.cdf(generator_X.sample((sample_size,)))
        return(data_x)