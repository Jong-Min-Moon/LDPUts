
    copula_mean_1 = -0.5 * torch.ones(d).to(device)
    copula_mean_2 =  -copula_mean_1


    coupla_sigma = (0.5 * torch.ones(d,d) + 0.5 * torch.eye(d)).to(device)

    server_private.load_private_data_multinomial(
        LDPclient.release_private_conti(
            priv_mech,
            data_gen.generate_copula_gaussian_data(sample_size, copula_mean_1, coupla_sigma),
            privacy_level,
            n_bin,
            device
        ),
        LDPclient.release_private_conti(
            priv_mech,
            data_gen.generate_copula_gaussian_data(sample_size, copula_mean_2, coupla_sigma),
            privacy_level,
            n_bin,
            device
        ),
    k,
    device,
    device
    )