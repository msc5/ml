def d4rl_sizes(env: str):

    if any([n in env for n in ('halfcheetah', 'walker')]):
        # return (17, 6)
        # return (9 + 9, 6)  # (qpos + qvel, action)
        return (17, 6)

    elif 'hopper' in env:
        # return (3, 11)
        return (6 + 6, 3)  # (qpos + qvel, action)
    elif 'ant' in env:
        return (28, 8)
        # return (15 + 14, 8)  # (qpos + qvel, action)
    elif 'door' in env:
        return (39, 28)
    elif 'hammer' in env:
        return (46, 26)
    elif 'kitchen' in env:
        return (60, 9)
    elif 'breakout' in env:
        return (1, 84, 84), 1
    else:
        raise NotImplementedError()
