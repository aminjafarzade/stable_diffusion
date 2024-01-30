import torch
import numpy as np

class DDIMSampler:

    def __init__(self, generator, num_training_steps=1000, beta_start=0.00085, beta_end=0.012):
        
        # timesteps tensor from n, n - 1, ..., 0
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

        # Noise Generator
        self.generator = generator

        self.num_train_steps = num_training_steps
        self.one = torch.tensor(1.0)
        # self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def set_strength(self, strength):
        
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step


    def set_inference_timesteps(self, n_inference_steps=50):
        self.num_inference_steps = n_inference_steps
        step_ratio = self.num_train_steps // n_inference_steps

        timesteps = (np.arange(0, n_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    
    def _get_previous_timestep(self, timestep):

        prev_time = timestep - (self.num_train_steps // self.num_inference_steps)
        return prev_time
    

    def _get_variance(self, timestep):

        t_prev = self._get_previous_timestep(timestep)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        var = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        # var = torch.clamp(var, min=1e-20)
        return var



    def add_noise(self, original_samples, timesteps):

        alpha_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(device=original_samples.device)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        one_minus_sqrt_alpha_prod = (1 - alpha_cumprod[timesteps]) ** 0.5
        one_minus_sqrt_alpha_prod = one_minus_sqrt_alpha_prod.flatten()
        while len(one_minus_sqrt_alpha_prod.shape) < len(original_samples.shape):
            one_minus_sqrt_alpha_prod = one_minus_sqrt_alpha_prod.unsqueeze(-1)

        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_prod * original_samples) + (noise * one_minus_sqrt_alpha_prod)

        return noisy_samples

        


    def step(self, timestep, latents, model_output, eta=0.0):
        t = timestep
        prev_t = self._get_previous_timestep(timestep)

        # if t > 0:  
            
        #     variance_noise = torch.randn(model_output.shape, generator=self.generator, device=model_output.shape, dtype=model_output.dtype)
        #     variance *= variance_noise
        
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one

        # beta_prod_t_prev = 1 - alpha_prod_t_prev
        beta_prod_t = 1 - alpha_prod_t

        # sqrt_alpha_prod_t = alpha_prod_t ** (0.5)
        # sqrt_alpha_prod_t_prev = alpha_prod_t_prev * (0.5)
        # sqrt_beta_prod_t = (1 - alpha_prod_t) ** (0.5)
        # sqrt_beta_prod_t_prev = (1 - alpha_prod_t_prev) ** (0.5)

        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        variance = self._get_variance(timestep)
        std_dev = eta * (variance ** (0.5))


        direction_pointer = (1 - alpha_prod_t_prev - std_dev ** 2) ** (0.5) * model_output

        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + direction_pointer

        if eta > 0:
            variance_noise = torch.randn(model_output.shape, generator=self.generator, device=model_output.shape, dtype=model_output.dtype)
            variance = std_dev * variance_noise

            prev_sample += variance

        return prev_sample
