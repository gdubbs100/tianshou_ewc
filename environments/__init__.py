from gymnasium.envs.registration import register

register(
    id="ModifiableCartPole-v0",
    entry_point="environments.custom_cartpole:ModifiableCartPole", # "module_name:ClassName"
    max_episode_steps=500,
)