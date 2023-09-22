import torch
import utils

from discretizer import discretizer

class client:
    def __init__(self, cuda_device):
        self.cuda_device = cuda_device
        self.discretizer = discretizer(cuda_device)
        self.LapU = LapU(cuda_device)
        self.discLapU = discLapU(cuda_device)
        self.genRR = genRR(cuda_device)
        self.bitFlip = bitFlip(cuda_device)
        

    def load_data_disc(self, data_y, data_z, alphabet_size):
        self.data_y = data_y
        self.data_z = data_z
        self.alphabet_size = alphabet_size

    def load_data_conti(self, data_y, data_z, n_bin):
        self.data_y, self.alphabet_size = self.discretizer.transform(data_y, n_bin)
        self.data_z, self.alphabet_size = self.discretizer.transform(data_z, n_bin)
    
    def release_LapU(self, privacy_level):
        return(
            self.LapU.privatize(self.data_y, self.alphabet_size, privacy_level),
            self.LapU.privatize(self.data_z, self.alphabet_size, privacy_level)
            )
    
    def release_DiscLapU(self, privacy_level):
        return(
            self.discLapU.privatize(self.data_y, self.alphabet_size, privacy_level),
            self.discLapU.privatize(self.data_z, self.alphabet_size, privacy_level)
            )

    def release_genRR(self, privacy_level):
        return(
            self.genRR.privatize(self.data_y, self.alphabet_size, privacy_level),
            self.genRR.privatize(self.data_z, self.alphabet_size, privacy_level)
               )

    def release_bitFlip(self, privacy_level):
        return(
            self.bitFlip.privatize(self.data_y, self.alphabet_size, privacy_level),
            self.bitFlip.privatize(self.data_z, self.alphabet_size, privacy_level)
               )

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
        zeta_alpha = torch.tensor(- privacy_level / (alphabet_size**(1/2))).exp().to(self.cuda_device)
        geometric_generator = torch.distributions.geometric.Geometric(1 - zeta_alpha)
        laplace_noise_disc  = geometric_generator.sample(sample_shape = torch.Size([sample_size, alphabet_size])).to(self.cuda_device)
        laplace_noise_disc = laplace_noise_disc.sub(geometric_generator.sample(sample_shape = torch.Size([sample_size, alphabet_size])).to(self.cuda_device))
        return(laplace_noise_disc)
     

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

        exp_alpha = torch.tensor(privacy_level).exp()
        p = torch.tensor([exp_alpha.divide(exp_alpha.add(1))]).to(self.cuda_device)
        bernoulli_dist = torch.distributions.bernoulli.Bernoulli(1-p) #0 value = stay, 1 value = flip
        bitFlipNoise = bernoulli_dist.sample((sample_size,alphabet_size)).view(sample_size, alphabet_size).to(self.cuda_device)

        data_flip = data_onehot.add(bitFlipNoise)
        data_flip = data_flip.add(
            data_flip.eq(2).mul(-2)
        )

        return(data_flip)  

