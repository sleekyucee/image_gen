import torch
import torch.nn as nn

class VAEEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super(VAEEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        self.fc_mean = nn.Linear(1024 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 4 * 4, latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mean(x)
        log_var = self.fc_logvar(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=128, in_channels=3):
        super(VAEDecoder, self).__init__()
        self.decoder_input = nn.Linear(latent_dim, 1024 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 1024, 4, 4)
        x = self.decoder(x)
        return x

class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = VAEDecoder(latent_dim, in_channels)

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var
