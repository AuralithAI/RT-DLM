import haiku as hk
import jax.numpy as jnp

class ImageGenerator(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.config = config
        assert isinstance(self.config.channels, int), "channels must be an integer"
        assert isinstance(self.config.latent_dim, int), "latent_dim must be an integer"

    def __call__(self, z):
        """Generate images from noise (latent vector)."""
        x = hk.Linear(1024)(z)  
        x = jnp.reshape(x, (-1, 16, 16, 4)) 
        x = hk.Conv2DTranspose(128, kernel_shape=4, stride=2, padding="SAME")(x)
        x = hk.Conv2DTranspose(64, kernel_shape=4, stride=2, padding="SAME")(x)
        x = hk.Conv2DTranspose(32, kernel_shape=4, stride=2, padding="SAME")(x)
        x = hk.Conv2DTranspose(int(self.config.channels), kernel_shape=4, stride=2, padding="SAME")(x)
        x = jnp.tanh(x)
        return x
    
def forward_fn(z, config):
    model = ImageGenerator(config)
    return model(z)

model = hk.transform_with_state(forward_fn)