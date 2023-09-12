class client:
    def __init__(self, privacy_level, cuda_device):
        self.privacy_level = privacy_level
        self.cuda_device = cuda_device
        self.discretizer = discretizer()
        self.LapU = LapU()
        self.discLapU = discLapU()

    def load_data_disc(self, data):
        self.data = data

    def load_data_conti(self, data):
        self.data = discretizer.transform(data)
    
    def release_LapU(self):
        return(self.LapU.privatize(self.data))
    
    def release_DiscLapU(self)
        return(self.discLapU.privatize(self.data))

   

    

