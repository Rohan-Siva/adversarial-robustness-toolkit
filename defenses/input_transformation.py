import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import config


def jpeg_compression(images, quality=None):
    
    if quality is None:
        quality = config.JPEG_QUALITY
    
    device = images.device
    batch_size = images.shape[0]
    compressed_images = torch.zeros_like(images)
    
    for i in range(batch_size):
                              
        img = images[i].cpu().detach()
        img = (img * 255).clamp(0, 255).byte()
        img = img.permute(1, 2, 0).numpy()
        
        if img.shape[2] == 1:
            img = img.squeeze(2)
            pil_img = Image.fromarray(img, mode='L')
        else:
            pil_img = Image.fromarray(img, mode='RGB')
        
                                
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        pil_img = Image.open(buffer)
        
                                
        img = np.array(pil_img)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        compressed_images[i] = img
    
    return compressed_images.to(device)


def bit_depth_reduction(images, bit_depth=None):
    
    if bit_depth is None:
        bit_depth = config.BIT_DEPTH
    
                                
    levels = 2 ** bit_depth
    reduced_images = torch.round(images * (levels - 1)) / (levels - 1)
    
    return reduced_images


def median_filter(images, kernel_size=3):
    
    device = images.device
    batch_size, channels, height, width = images.shape
    filtered_images = torch.zeros_like(images)
    
    pad = kernel_size // 2
    
    for i in range(batch_size):
        for c in range(channels):
            img = images[i, c].cpu().numpy()
            
                       
            padded = np.pad(img, pad, mode='reflect')
            
                                 
            filtered = np.zeros_like(img)
            for h in range(height):
                for w in range(width):
                    window = padded[h:h+kernel_size, w:w+kernel_size]
                    filtered[h, w] = np.median(window)
            
            filtered_images[i, c] = torch.from_numpy(filtered)
    
    return filtered_images.to(device)


def gaussian_noise(images, std=0.01):
    
    noise = torch.randn_like(images) * std
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0, 1)
    
    return noisy_images


def spatial_smoothing(images, kernel_size=3):
    
    pad = kernel_size // 2
    smoothed = F.avg_pool2d(
        F.pad(images, (pad, pad, pad, pad), mode='reflect'),
        kernel_size=kernel_size,
        stride=1
    )
    
    return smoothed


def total_variance_minimization(images, weight=0.1, iterations=10):
    
    device = images.device
    denoised = images.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([denoised], lr=0.01)
    
    for _ in range(iterations):
        optimizer.zero_grad()
        
                            
        data_loss = F.mse_loss(denoised, images)
        
                              
        tv_h = torch.abs(denoised[:, :, 1:, :] - denoised[:, :, :-1, :]).sum()
        tv_w = torch.abs(denoised[:, :, :, 1:] - denoised[:, :, :, :-1]).sum()
        tv_loss = tv_h + tv_w
        
                    
        loss = data_loss + weight * tv_loss
        
        loss.backward()
        optimizer.step()
        
                              
        with torch.no_grad():
            denoised.clamp_(0, 1)
    
    return denoised.detach()


class InputTransformationDefense:
    
    
    def __init__(self, transformations):
        
        self.transformations = transformations
    
    def __call__(self, images):
        
        transformed = images
        for transform in self.transformations:
            transformed = transform(transformed)
        return transformed
    
    def defend_model(self, model):
        
        class DefendedModel(torch.nn.Module):
            def __init__(self, base_model, defense):
                super().__init__()
                self.base_model = base_model
                self.defense = defense
            
            def forward(self, x):
                x = self.defense(x)
                return self.base_model(x)
        
        return DefendedModel(model, self)
