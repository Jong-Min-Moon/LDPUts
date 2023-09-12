class LapU:
    def __init__(self, cuda_device):
        self.cuda_device = cuda_device
        self.initialize_laplace_generator()


    def privatize(self, data_mutinomial, alphabet_size, privacy_level):
        data_onehot = self.transform_onehot(data_mutinomial, alphabet_size)
        laplace_scale = torch.tensor(2*alphabet_size).sqrt().mul(2).div(privacy_level).to(self.cuda_device) #sigma_alpha in the paper
        sample_size = self.get_sample_size(data_mutinomial).to(self.cuda_device)
        laplace_noise = self.generate_random_noise(sample_size, alphabet_size).to(self.cuda_device)
        return(
            torch.add(
                data_onehot.mul(sqrt(alphabet_size)),
                laplace_noise.mul(laplace_scale)
            )  
        )
    
    def generate_random_noise(self, sample_size, alphabet_size):
        self.unit_laplace_generator.sample(sample_shape = sample_size.mul(alphabet_size))

    def LapU(self, oneHot, alpha, c, theta):
        p = torch.exp(torch.tensor(
            - alpha / (c * theta)
            )).to(self.cuda_device)
        laplaceSize = oneHot.size()
        laplaceNoise = self.generate_disc_laplace(p, laplaceSize)
        LDPView = torch.tensor(theta) * oneHot + laplaceNoise
        return(LDPView)
    
    def transform_onehot(data_multinomial, alphabet_size)
        return(torch.nn.functional.one_hot(data_multinomial, alphabet_size))

    def get_sample_size(self, data):
        if data.dim() == 1:
            return( data.size(dim = 0) )
        elif data.dim() == 2:
            return( data.size(dim = 1) )
        else:
            return # we only use up to 2-dimensional tensor, i.e. matrix
        
    def initialize_laplace_generator(self)
        self.unit_laplace_generator = torch.distributions.laplace.Laplace(
            torch.tensor(0.0).to(self.cuda_device),
            torch.tensor(2**(-1/2)).to(self.cuda_device)
        )