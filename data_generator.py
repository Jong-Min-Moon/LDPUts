import torch


class purturbed_unif_p:

    def __init__(self, alphabet_size, bump_size, privacy_level):
        self.p1, self.p2 = self.get_p(alphabet_size, bump_size)
        self.squred_eltwo = self.get_squared_elltwo(self.p1, self.p2)
        self.alpha_bf, self.delta_bf = self.get_bitflip_params(privacy_level)
        self.expectation_bitflip_elltwo = self.alpha_bf.square().mul(self.squred_eltwo)

    def get_p(self, alphabet_size, bump_size):
        p = torch.ones(alphabet_size).div(alphabet_size)
        p2 = p.add(
                torch.remainder(
                torch.tensor(range(alphabet_size)),
                2
                ).add(-1/2).mul(2).mul(bump_size)
                )
        p1_idx = torch.cat( ( torch.arange(1, alphabet_size), torch.tensor([0])), 0)
        p1 = p2[p1_idx]
        return(p1, p2)

    def get_squared_elltwo(self, p1, p2):
        squared_elltwo = torch.sum(torch.pow(torch.subtract(p1, p2), 2), dim=0)
        return(squared_elltwo)
        
    def get_bitflip_params(self, privacy_level):
        alpha_bf = torch.tensor(privacy_level).div(2).exp().sub(1).div(
            torch.tensor(privacy_level).div(2).exp().add(1)
        )
        delta_bf = torch.tensor(privacy_level).div(2).exp().add(1).reciprocal()
        return(alpha_bf, delta_bf)

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