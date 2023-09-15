def generate_nearly_unif(alphabet_size, beta, sample_size):
  p = torch.arange(1,alphabet_size+1).pow(-beta)
  p = p.divide(p.sum())
  return(torch.multinomial(p, sample_size, replacement=True))