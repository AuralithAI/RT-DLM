class ImageGenConfig:
    def __init__(self, image_size=256, channels=3, latent_dim=128):
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.latent_dim = int(latent_dim)