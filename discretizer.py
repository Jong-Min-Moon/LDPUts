class discretizer:
    def __init__(self, cuda_device):
        self.cuda_device = cuda_device

    def transform(self, data, n_bin): 
        ''' 
        input arguments
            data: 2d torch tensor of continuous data
            n_bin: number of bin in each dimension
        output
            torch tensor of multivariate data
        '''          
        # create designated number of intervals
        data_bin_index = self.transform_bin_index(data, n_bin) # each column into bin index
        data_multinomial = self.transform_multinomial(data_bin_index, n_bin) # all column in to a single column with n_bin^d categories  
        return(data_multinomial)
    
    def get_dimension(self, data):
        if data.dim() == 1:
            return(1)
        elif data.dim() == 2:
            return( data.size(dim = 1) )
        else:
            return # we only use up to 2-dimensional tensor, i.e. matrix
    
    def transform_bin_index(self, data, n_bin):
        '''
        for each dimension, transform the data in [0,1] into the interval index
        first interval = [0, x], the others = (y z]
        
        input: two
            1. data: torch tensor object on GPU
            2. n_bin: integer
        output: one
            1. bin_index: torch tensor of bin indices, dimension same as the input
        '''      
        bin = torch.linspace(start = 0, end = 1, steps = n_bin + 1).to(self.cuda_device) #create bins (floatTensor)
        bin_index = torch.bucketize(data, bin, right = False) # bin index
        bin_index = bin_index.add(bin_index.eq(0)) #move 0 values from the bin number 0 to the bin number 1       
        return(bin_index)    

    def transform_multinomial(self, data_bin_index, n_bin):
        """Only for continuous and multivariate data ."""
        d = self.get_dimension(data_bin_index)
        if d == 1:
            return(data_bin_index.sub(1).reshape(-1,))
        else:
            exponent = torch.linspace(start = (d-1), end = 0, steps = d, dtype = torch.long)
            vector = torch.tensor(n_bin).pow(exponent)
            return( torch.matmul( data_bin_index.sub(1).to(torch.float), vector.to(torch.float).to(self.cuda_device) ).to(torch.long) )   
    
    