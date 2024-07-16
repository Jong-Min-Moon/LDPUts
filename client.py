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

    def release_private_conti(self, method_name, data, privacy_level, n_bin, cuda_device):
        if method_name == "lapu":
            return( self.lapu.privatize_conti(data, privacy_level, n_bin, cuda_device) )
        elif method_name == "genrr":
            return( self.genrr.privatize_conti(data, privacy_level, n_bin, cuda_device) )
        elif method_name =="bitflip":
            return( self.bitflip.privatize_conti(data, privacy_level, n_bin, cuda_device) )
            



    
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
        # privacy_level is unused intentionally.
        laplace_noise = self.unit_laplace_generator.sample(sample_shape = torch.Size([sample_size, alphabet_size]))
        return(laplace_noise)

    def privatize_conti(self, data_conti, privacy_level, n_bin, cuda_device):
        self.discretizer = discretizer(cuda_device)
        data_multinomial, alphabet_size = self.discretizer.transform(data_conti, n_bin)
        data_private = self.privatize(data_multinomial, alphabet_size, privacy_level, cuda_device)
        return(data_private)

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
        flip_probability = torch.tensor(privacy_level).div(2).exp().reciprocal()
        random_mask = torch.bernoulli(torch.full((sample_size, alphabet_size), flip_probability)).int().bool()
        flipped_matrix = torch.nn.functional.one_hot(data_mutinomial, alphabet_size)
        flipped_matrix[random_mask] = 1 - flipped_matrix[random_mask]
        return(flipped_matrix)