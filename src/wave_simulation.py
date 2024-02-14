import torch


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
        # Determine the best device (CPU or GPU via MPS)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.global_dampening = 1.0
        self.source_opacity = 0.9
        self.c = torch.ones((h, w), dtype=torch.float32, device=self.device)  # wave speed field
        self.d = torch.ones((h, w), dtype=torch.float32, device=self.device)  # dampening field
        self.u = torch.zeros((h, w), dtype=torch.float32, device=self.device)  # field values, amplitude of each pixel
        self.u_prev = torch.zeros((h, w), dtype=torch.float32, device=self.device)  # field values of previous frame

        self.set_dampening_field(None, 32)

        # Define Laplacian kernel
        self.laplacian_kernel = torch.tensor([[0.05, 0.2, 0.05],
                                              [0.2, -1.0, 0.2],
                                              [0.05, 0.2, 0.05]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        self.t = 0
        self.dt = 1.0
        self.sources = torch.zeros([0, 5], dtype=torch.float32, device=self.device)

    def reset_time(self):
        self.t = 0.0

    def update_field(self):
        # Perform 2D convolution for the laplacian
        # unsqueeze() twice for [1, 1, h, w] as [batch_size, channels, height, width]
        laplacian = torch.nn.functional.conv2d(self.u.unsqueeze(0).unsqueeze(0), self.laplacian_kernel, padding='same').squeeze()

        # Update field
        v = (self.u - self.u_prev) * self.d  # velocity of sound
        r = (self.u + v + laplacian * (self.c * self.dt)**2)  # new state

        self.u_prev[:] = self.u
        self.u[:] = r

        self.t += self.dt

    def get_field(self):
        return self.u.cpu().numpy()

    def set_dampening_field(self, d, pml_thickness):
        # Check if d is already a tensor and ensure it's on the correct device and has the correct dtype
        if isinstance(d, torch.Tensor):
            # If d is a tensor, directly adjust its device and dtype without cloning if it's already correct
            d_tensor = d.to(dtype=torch.float32, device=self.device)
        else:
            # If d is not a tensor (e.g., a numpy array or None), create a new tensor
            if d is not None:
                d_tensor = torch.tensor(d, dtype=torch.float32, device=self.device)
            else:
                # If d is None, create a tensor filled with global_dampening values
                d_tensor = torch.full((self.d.shape[0], self.d.shape[1]), self.global_dampening, dtype=torch.float32,
                                      device=self.device)

        # Apply clipping to ensure dampening values are within valid range
        self.d = torch.clip(d_tensor, 0.0, self.global_dampening)

        # Adjust dampening at the edges based on pml_thickness to prevent reflections
        w, h = self.d.shape[1], self.d.shape[0]
        for i in range(pml_thickness):
            v = ((i + 1) / pml_thickness) ** 0.5
            self.d[i, i:w - i] = v
            self.d[h - (i + 1), i:w - i] = v
            self.d[i:h - i, i] = v
            self.d[i:h - i, w - (i + 1)] = v

    def set_refractive_index_field(self, r):
        if isinstance(r, torch.Tensor):
            # If r is already a tensor, adjust device and dtype as needed without cloning
            r_tensor = r.to(dtype=torch.float32, device=self.c.device)
        else:
            # If r is not a tensor (e.g., a numpy array), create a new tensor
            r_tensor = torch.tensor(r, dtype=torch.float32, device=self.c.device)
        self.c = 1.0 / torch.clip(r_tensor, 1.0, 10.0)

    def set_sources(self, sources):
        self.sources = torch.tensor(sources, dtype=torch.float32, device=self.c.device)

    def update_sources(self):
        v = torch.sin(self.sources[:, 2] + self.sources[:, 4] * self.t) * self.sources[:, 3]
        coords = self.sources[:, :2].long()

        for i, (x, y) in enumerate(coords):
            self.u[y, x] = self.u[y, x] * self.source_opacity + v[i] * (1.0 - self.source_opacity)

