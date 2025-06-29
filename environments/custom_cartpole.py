from gymnasium.envs.classic_control import CartPoleEnv

## create custom cartpole
class ModifiableCartPole(CartPoleEnv):
    """
    CartPole environment with customisable inputs to change dynamics
    """
    def __init__(self, 
                gravity:float=9.8, 
                masscart:float=1.0,
                masspole:float=0.1,
                length:float=0.5,
                force_mag:float=10.0,
                **kwargs
        ):
        super().__init__(**kwargs)
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = self.masspole + self.masscart
        self.length = length  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = force_mag

