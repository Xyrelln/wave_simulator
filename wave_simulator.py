import torch
import torch.nn.functional as F

class WaveSimulator2D:
    """
    Simulates the 2D wave equation on Apple Silicon GPU using PyTorch
    The system assumes units, where the wave speed is 1.0 pixel/timestep
    source frequency should be adjusted accordingly
    """
    def __init__(self, w, h):
        """
        Initialize the 2D wave simulator.
        @param w: Width of the simulation grid.
        @param h: Height of the simulation grid.
        """
        device = torch.device("mps")  # Target Apple Silicon GPU

        self.global_dampening = 1.0
        self.source_opacity = 0.9
        self.c = torch.ones((h, w), dtype=torch.float32, device=device)  # wave speed field
        self.d = torch.ones((h, w), dtype=torch.float32, device=device)  # dampening field
        self.u = torch.zeros((h, w), dtype=torch.float32, device=device)  # field values
        self.u_prev = torch.zeros((h, w), dtype=torch.float32, device=device)  # field values of previous frame

        self.set_dampening_field(None, 32)

        # Define Laplacian kernel
        self.laplacian_kernel = torch.tensor([[0.05, 0.2, 0.05],
                                              [0.2, -1.0, 0.2],
                                              [0.05, 0.2, 0.05]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        self.t = 0
        self.dt = 1.0
        self.sources = torch.zeros([0, 5], dtype=torch.float32, device=device)

    def reset_time(self):
        self.t = 0.0

    def update_field(self):
        # Perform 2D convolution for the laplacian
        laplacian = F.conv2d(self.u.unsqueeze(0).unsqueeze(0), self.laplacian_kernel, padding='same').squeeze()

        # Update field
        v = (self.u - self.u_prev) * self.d
        r = (self.u + v + laplacian * (self.c * self.dt)**2)

        self.u_prev[:] = self.u
        self.u[:] = r

        self.t += self.dt

    def get_field(self):
        return self.u.cpu().numpy()

    def set_dampening_field(self, d, pml_thickness):
        if d is not None:
            self.d = torch.clip(torch.tensor(d, device=self.d.device), 0.0, self.global_dampening)
        else:
            self.d.fill_(self.global_dampening)

        # Apply dampening at the boundaries
        w, h = self.d.shape[1], self.d.shape[0]
        for i in range(pml_thickness):
            v = (i / pml_thickness) ** 0.5
            self.d[i, i:w-i] = v
            self.d[h-(1+i), i:w-i] = v
            self.d[i:h-i, i] = v
            self.d[i:h-i, w-(1+i)] = v

    def set_refractive_index_field(self, r):
        self.c = 1.0 / torch.clip(torch.tensor(r, device=self.c.device), 1.0, 10.0)

    def set_sources(self, sources):
        self.sources = torch.tensor(sources, dtype=torch.float32, device=self.c.device)

    def update_sources(self):
        v = torch.sin(self.sources[:, 2] + self.sources[:, 4] * self.t) * self.sources[:, 3]
        coords = self.sources[:, :2].long()

        for i, (x, y) in enumerate(coords):
            self.u[y, x] = self.u[y, x] * self.source_opacity + v[i] * (1.0 - self.source_opacity)

