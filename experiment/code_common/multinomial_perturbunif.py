
    p = torch.ones(d).div(d)
    p2 = p.add(
        torch.remainder(
        torch.tensor(range(d)),
        2
        ).add(-1/2).mul(2).mul(bump)
    )
    p1_idx = torch.cat( ( torch.arange(1, d), torch.tensor([0])), 0)
    p1 = p2[p1_idx]
    

    server_private.load_private_data_multinomial(
        LDPclient.release_private(
            priv_mech,
            data_gen.generate_multinomial_data(p1, sample_size),
            k,
            privacy_level,
            device
        ),
        LDPclient.release_private(
            priv_mech,
            data_gen.generate_multinomial_data(p2, sample_size),
            k,
            privacy_level,
            device
        ),
    d,
    device,
    device
    )
         