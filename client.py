import torch
import utils

from discretizer import discretizer

class client:
    def __init__(self, cuda_device, privacy_level):
        self.cuda_device = cuda_device
        self.privacy_level = privacy_level
        
        self.discretizer = discretizer(cuda_device)
        self.LapU = LapU(cuda_device)
        self.discLapU = discLapU(cuda_device)
        self.genRR = genRR(cuda_device)
        self.bitFlip = bitFlip(cuda_device)
        self.geomU = geomU(cuda_device)
        self.discLapU_test = DiscLapU_test(cuda_device)
        
    def load_data_multinomial(self, data_y, data_z, alphabet_size):
        self.data_y = data_y
        self.data_z = data_z
        self.alphabet_size = alphabet_size

    def load_data_conti(self, data_y, data_z, n_bin):
        self.data_y, self.alphabet_size = self.discretizer.transform(data_y, n_bin)
        self.data_z, self.alphabet_size = self.discretizer.transform(data_z, n_bin)
    
    def release_LapU(self):
        return(
            self.LapU.privatize(self.data_y, self.alphabet_size, self.privacy_level),
            self.LapU.privatize(self.data_z, self.alphabet_size, self.privacy_level)
            )
    
    def release_DiscLapU(self):
        return(
            self.discLapU.privatize(self.data_y, self.alphabet_size, self.privacy_level),
            self.discLapU.privatize(self.data_z, self.alphabet_size, self.privacy_level)
            )

    def release_DiscLapU_test(self):
        return(
            self.discLapU_test.privatize(self.data_y, self.alphabet_size, self.privacy_level),
            self.discLapU_test.privatize(self.data_z, self.alphabet_size, self.privacy_level)
            )
    def release_geomU(self):
        return(
            self.geomU.privatize(self.data_y, self.alphabet_size, self.privacy_level),
            self.geomU.privatize(self.data_z, self.alphabet_size, self.privacy_level)
            )
    def release_genRR(self):
        return(
            self.genRR.privatize(self.data_y, self.alphabet_size, self.privacy_level),
            self.genRR.privatize(self.data_z, self.alphabet_size, self.privacy_level),
               )

    def release_bitFlip(self):
        return(
            self.bitFlip.privatize(self.data_y, self.alphabet_size, self.privacy_level),
            self.bitFlip.privatize(self.data_z, self.alphabet_size, self.privacy_level)
               )

class DiscLapU_test:
    def __init__(self, cuda_device):
        self.cuda_device = cuda_device

    def privatize(self, data_mutinomial, alphabet_size, privacy_level):
        sample_size = utils.get_sample_size(data_mutinomial)
        data_onehot_scaled = torch.nn.functional.one_hot(data_mutinomial, alphabet_size)#.mul(alphabet_size**(1/2)) # scaled by \sqrt(k)
        noise = self._generate_noise(alphabet_size, privacy_level, sample_size)
        return(torch.add(data_onehot_scaled,noise))
    
    def _generate_noise(self, alphabet_size, privacy_level, sample_size):
        zeta_alpha = torch.tensor(- privacy_level).div(2).div(alphabet_size**(1/2)).exp().to(self.cuda_device)
        geometric_generator = torch.distributions.geometric.Geometric(1 - zeta_alpha)
        laplace_noise_disc  = geometric_generator.sample(sample_shape = torch.Size([sample_size, alphabet_size])).to(self.cuda_device)
        laplace_noise_disc = laplace_noise_disc.sub(geometric_generator.sample(sample_shape = torch.Size([sample_size, alphabet_size])).to(self.cuda_device))
        return(laplace_noise_disc)
    
class LapU:
    def __init__(self, cuda_device):
        self.cuda_device = cuda_device
        self._initialize_laplace_generator()

    def privatize(self, data_mutinomial, alphabet_size, privacy_level):
        sample_size = utils.get_sample_size(data_mutinomial)
        data_onehot_scaled = torch.nn.functional.one_hot(data_mutinomial, alphabet_size).mul(alphabet_size**(1/2)) # scaled by \sqrt(k)
        noise = self._generate_noise(alphabet_size, privacy_level, sample_size)
        return(torch.add(data_onehot_scaled,noise))
           
    def _generate_noise(self, alphabet_size, privacy_level, sample_size):
        laplace_scale = torch.tensor(2*alphabet_size).sqrt().mul(2).div(privacy_level).to(self.cuda_device) #sigma_alpha in the paper
        laplace_noise = self.unit_laplace_generator.sample(sample_shape = torch.Size([sample_size, alphabet_size])).to(self.cuda_device)
        return(laplace_noise.mul(laplace_scale))

        

    def _get_sample_size(self, data):
        if data.dim() == 1:
            return( data.size(dim = 0) )
        elif data.dim() == 2:
            return( data.size(dim = 1) )
        else:
            return # we only use up to 2-dimensional tensor, i.e. matrix 
        
    def _initialize_laplace_generator(self):
        self.unit_laplace_generator = torch.distributions.laplace.Laplace(
            torch.tensor(0.0).to(self.cuda_device),
            torch.tensor(2**(-1/2)).to(self.cuda_device)
        )


class discLapU(LapU):
    def _generate_noise(self, alphabet_size, privacy_level, sample_size):
        zeta_alpha = torch.tensor(- privacy_level).div(2).div(alphabet_size**(1/2)).exp().to(self.cuda_device)
        geometric_generator = torch.distributions.geometric.Geometric(1 - zeta_alpha)
        laplace_noise_disc  = geometric_generator.sample(sample_shape = torch.Size([sample_size, alphabet_size])).to(self.cuda_device)
        laplace_noise_disc = laplace_noise_disc.sub(geometric_generator.sample(sample_shape = torch.Size([sample_size, alphabet_size])).to(self.cuda_device))
        return(laplace_noise_disc)

class geomU(LapU):
    def _generate_noise(self, alphabet_size, privacy_level, sample_size):
        geom_param = self.geom(privacy_level)
        geometric_generator = torch.distributions.geometric.Geometric(geom_param)
        noise = geometric_generator.sample(sample_shape = torch.Size([sample_size, alphabet_size])).to(self.cuda_device)
        mean = geom_param.mul(-1).add(1).div(geom_param)
        return(noise.sub(mean).mul(alphabet_size**(1/2)))
    
    def geom(self, alpha):

        
        root= torch.tensor(alpha).exp().sub(
            torch.tensor(2*alpha).exp().sub(4).sqrt()
        ).div(2)
        return (1-root)

class genRR(LapU):   
    def privatize(self, data_mutinomial, alphabet_size, privacy_level):
        sample_size = utils.get_sample_size(data_mutinomial)
        data_onehot = torch.nn.functional.one_hot(data_mutinomial, alphabet_size)
        one_matrix = torch.zeros(size = torch.Size([sample_size, alphabet_size])).add(1).to(self.cuda_device)
        bias = torch.tensor(privacy_level).exp()
        bias_matrix = data_onehot.mul(bias).add(one_matrix).sub(data_onehot)

        p = 1 / ( torch.tensor(privacy_level).exp().add(alphabet_size - 1) )
        p = torch.zeros(size = torch.Size([sample_size, alphabet_size])).add(1).mul(p).to(self.cuda_device)
        p = p.mul(bias_matrix)
        random_multinomial = torch.multinomial(p, 1).view(-1).to(self.cuda_device)
        return(random_multinomial)  

class bitFlip(LapU):   
    def privatize(self, data_mutinomial, alphabet_size, privacy_level):
        """
        output: bit vector in (0,1)^k
        """
        sample_size = utils.get_sample_size(data_mutinomial)
        data_onehot = torch.nn.functional.one_hot(data_mutinomial, alphabet_size)

        alpha_half = torch.tensor(privacy_level).div(2)

        log_p = alpha_half - alpha_half.exp().add(1).log().to(self.cuda_device)
        p = log_p.exp()
        bernoulli_dist = torch.distributions.bernoulli.Bernoulli(1-p) #0 value = stay, 1 value = flip
        bitFlipNoise = bernoulli_dist.sample((sample_size,alphabet_size)).view(sample_size, alphabet_size).to(self.cuda_device)

        data_flip = data_onehot.add(bitFlipNoise)
        data_flip = data_flip.add(
            data_flip.eq(2).mul(-2)
        )

        return(data_flip)