from gymnasium.envs.registration import register

register(
    id="myCartPole-v0",
    entry_point="env.my_cartpole:myCartPoleEnv",
    )

register(
    id="BRC-v0",
    entry_point="env.brc_env:BRCEnv",
    )