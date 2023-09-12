class laplace_generator:
    def __init__(self, cuda_device):
        self.cuda_device = cuda_device
        self.initialize_laplace_generator()

def generate_unit_laplace(self, k):
        '''
        k: torch.size object with size 1 * alphabet size
        output: torch tensor of data from unit laplace distribution
        '''
        return self.unit_laplace_generator.sample(sample_shape = k)



def release_LapU(self, alpha):
        ''' Only for continuous data.
        for each dimension, transform the data in [0,1] into the interval index
        first interval = [0, x], the others = (y z]
        
        input arguments
            alpha: privacy level
        output
            LDPView: \alpha-LDP view of the input multivariate data
        '''
        sigma = torch.tensor(2).sqrt().div(alpha)
        laplaceNoise = self.generate_unit_laplace(self.discdata.size())
        LDPView =  torch.add(
            self.discdata.mul(self.discdata.),
            laplaceNoise.mul(sigma)
        )
        return(LDPView)