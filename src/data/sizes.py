def d4rl_sizes(env: str):

    # if 'walker' in env:
    #     return (17, 6)
    # elif 'halfcheetah' in env:
    #     return (17, 6)
    # elif 'hopper' in env:
    #     return (11, 3)
    # elif 'maze2d' in env:
    #     return (4, 2)
    #
    # elif 'ant' in env:
    #     return (28, 8)
    #
    # elif 'door' in env:
    #     return (39, 28)
    # elif 'hammer' in env:
    #     return (46, 26)
    # elif 'kitchen' in env:
    #     return (60, 9)
    # elif 'breakout' in env:
    #     return (1, 84, 84), 1
    # else:
    #     raise NotImplementedError()

    # env name -> (x_size, a_size)
    sizes = {'walker': (17, 6),
             'halfcheetah': (17, 6),
             'hopper': (11, 3),
             'ant': (28, 8),
             'maze2d': (4, 2)}

    # size = sizes.get(env, None)
    for name in sizes:
        if name in env:
            return sizes[name]

    raise NotImplementedError()
