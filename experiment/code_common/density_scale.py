    
    copula_mean_1 = torch.zeros(d).to(device)
    copula_mean_2 = copula_mean_1

    copula_sigma_1 = (0.5 * torch.ones(d,d) + 0.5 * torch.eye(d)).to(device)
    copula_sigma_2 = copula_sigma_1.mul(5)
    
    server_private.load_private_data_multinomial(
        LDPclient.release_private_conti(
            priv_mech,
            data_gen.generate_copula_gaussian_data(sample_size, copula_mean_1, copula_sigma_1),
            privacy_level,
            n_bin,
            device
        ),
        LDPclient.release_private_conti(
            priv_mech,
            data_gen.generate_copula_gaussian_data(sample_size, copula_mean_2, copula_sigma_2),
            privacy_level,
            n_bin,
            device
        ),
    k,
    device,
    device
    )