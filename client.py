class client:
    def __init__(self, privacy_level, cuda_device):
        self.privacy_level = privacy_level
        self.cuda_device = cuda_device
        self.discretizer = discretizer(cuda_device)
        self.LapU = LapU(cuda_device)
        self.discLapU = discLapU(cuda_device)

    def load_data_disc(self, data, alphabet_size):
        self.data = data
        self.alphabet_size = alphabet_size

    def load_data_conti(self, data, n_bin):
        self.data, self.alphabet_size = self.discretizer.transform(data, n_bin)
    
    def release_LapU(self):
        return(self.LapU.privatize(self.data), self.alphabet_size, self.privacy_level)
    
    def release_DiscLapU(self):
        return(self.discLapU.privatize(self.data), self.alphabet_size, self.privacy_level)

   

    

