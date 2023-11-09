import torch

class data_generator:
    def __init__(self, cuda_device):
        self.cuda_device = cuda_device

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
            ).to(self.cuda_device)
        )
    def _generate_power_law_p(self, alphabet_size, beta):
        p = torch.arange(1,alphabet_size+1).pow(-beta).to(self.cuda_device)
        p = p.divide(p.sum())
        return(p)
    
    def generate_power_law_p_private(self, alphabet_size, beta, privacy_level):
        p = torch.arange(1,alphabet_size+1).pow(-beta).to(self.cuda_device)
        p = p.divide(p.sum())
        exp_alpha = torch.tensor(privacy_level).exp()
        denumerator = exp_alpha.add(alphabet_size).sub(1)
        p_private = (p.mul(exp_alpha) + (1-p)).div(denumerator)
        
        return(p, p_private)