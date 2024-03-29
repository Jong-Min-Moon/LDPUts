import torch
import utils

from discretizer import discretizer

class client:
    def __init__(self):
        
        self.lapu = lapu()
        self.disclapu = disclapu()
        self.genrr = genrr()
        self.bitflip = bitflip()

    
    def release_private(self, method_name, data, alphabet_size, privacy_level, cuda_device):
        if method_name == "lapu":
            return( self.lapu.privatize(data, alphabet_size, privacy_level, cuda_device) )
        elif method_name == "genrr":
            return( self.genrr.privatize(data, alphabet_size, privacy_level, cuda_device) )
        elif method_name =="bitflip":
            return( self.bitflip.privatize(data, alphabet_size, privacy_level, cuda_device) )
    



        


        
  

    
""" class truncGaussU:
    def __init__(self ):


    def privatize(self, data_mutinomial, alphabet_size, privacy_level):
        sample_size = utils.get_sample_size(data_mutinomial)
        data_onehot = torch.nn.functional.one_hot(data_mutinomial, alphabet_size)
        noise = self._generate_noise(alphabet_size, privacy_level, sample_size)
        return(
            data_onehot.add(noise).mul(alphabet_size**0.5)
        )
    
    def _generate_noise(self, alphabet_size, privacy_level, sample_size):
        noise = torch.empty(sample_size, alphabet_size)
        upper_limit = (1/privacy_level - 1/2)
        noise = torch.nn.init.trunc_normal_(noise,  mean=0.0, std=(2**0.5)/privacy_level, a = -upper_limit, b=upper_limit)
        return(noise) """
    
class lapu:
    def __init__(self):
        self._initialize_laplace_generator()

    def privatize(self, data_mutinomial, alphabet_size, privacy_level, cuda_device):
        sample_size = utils.get_sample_size(data_mutinomial)
        data_private = torch.nn.functional.one_hot(data_mutinomial, alphabet_size).add(
            self._generate_noise(alphabet_size, privacy_level, sample_size).mul(
                torch.tensor(8**0.5, dtype=torch.float32)
                ).div(privacy_level)
        ).mul(
            torch.tensor(alphabet_size, dtype=torch.float32).sqrt()
        )
        return(data_private.to(cuda_device))
           
    def _generate_noise(self, alphabet_size, privacy_level, sample_size):
        laplace_noise = self.unit_laplace_generator.sample(sample_shape = torch.Size([sample_size, alphabet_size]))
        return(laplace_noise)

        

    def _get_sample_size(self, data):
        if data.dim() == 1:
            return( data.size(dim = 0) )
        elif data.dim() == 2:
            return( data.size(dim = 1) )
        else:
            return # we only use up to 2-dimensional tensor, i.e. matrix 
        
    def _initialize_laplace_generator(self):
        self.unit_laplace_generator = torch.distributions.laplace.Laplace(
            torch.tensor(0.0),
            torch.tensor(2**(-1/2))
        )


class disclapu(lapu):
    def _generate_noise(self, alphabet_size, privacy_level, sample_size):
        zeta_alpha = torch.tensor(- privacy_level).div(2).div(alphabet_size**(1/2)).exp()
        geometric_generator = torch.distributions.geometric.Geometric(1 - zeta_alpha)
        laplace_noise_disc  = geometric_generator.sample(sample_shape = torch.Size([sample_size, alphabet_size]))
        laplace_noise_disc = laplace_noise_disc.sub(geometric_generator.sample(sample_shape = torch.Size([sample_size, alphabet_size])))
        return(laplace_noise_disc)


class genrr(lapu):   
    def privatize(self, data_mutinomial, alphabet_size, privacy_level, cuda_device):
        privacy_level_exp = torch.tensor(privacy_level, dtype=torch.float64).exp()
        sample_size = utils.get_sample_size(data_mutinomial)
        data_onehot = torch.nn.functional.one_hot(data_mutinomial, alphabet_size)
        one_matrix = torch.zeros(size = torch.Size([sample_size, alphabet_size])).add(1)

        bias_matrix = data_onehot.mul(
            privacy_level_exp
            ).add(one_matrix).sub(data_onehot)

        p = 1 / ( privacy_level_exp.add(alphabet_size - 1) )
        p = torch.zeros(size = torch.Size([sample_size, alphabet_size])).add(1).mul(p)
        p = p.mul(bias_matrix)
        return( torch.multinomial(p, 1).view(-1))  

class bitflip(lapu):   
    def privatize(self, data_mutinomial, alphabet_size, privacy_level, cuda_device):
        """
        output: bit vector in (0,1)^k
        """
        sample_size = utils.get_sample_size(data_mutinomial)

        alpha_half = torch.tensor(privacy_level).div(2)

        log_p = alpha_half - alpha_half.exp().add(1).log()
        p = log_p.exp()
        bernoulli_dist = torch.distributions.bernoulli.Bernoulli(1-p) #0 value = stay, 1 value = flip


        data_bitflip = torch.nn.functional.one_hot(data_mutinomial, alphabet_size).add( #one-hottize
            bernoulli_dist.sample((sample_size,alphabet_size)).view(sample_size, alphabet_size)
        )
        data_bitflip = data_bitflip.add(
            data_bitflip.eq(2).mul(-2)
        )

        return(data_bitflip)