import torch
import torch.nn as nn

class AdaIR_SR(nn.Module):
    def __init__(self, pretrained_path=None, in_channels=1, upscale=2):
        super(AdaIR_SR, self).__init__()
        self.upscale = upscale
        
        # For the hackathon, we assume AdaIR backbone expects 3 channels.
        # This wrapper repeats the single channel to 3 channels to use the pretrained backbone safely
        # without randomly reinitializing the first layer weights.
        
        try:
            # Placeholder for actual AdaIR import (this path depends on the cloned repo's structure)
            from adair.basicsr.models.archs.adair_arch import AdaIR
            self.backbone = AdaIR(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4)
        except ImportError:
            # Mock backbone for testing scripts if AdaIR isn't downloaded yet
            print("Warning: AdaIR module not found. Using a dummy CNN backbone for structural tests.")
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 48, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(48, 3, 3, 1, 1)
            )

        # PixelShuffle SR Head
        # Output features from AdaIR -> upscale factor
        # Since input to pixel shuffle needs to be in_channels * (upscale ** 2), which is 1 * 4 = 4
        self.upsampler = nn.Sequential(
            nn.Conv2d(3, in_channels * (upscale ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale)
        )
        
        # Load Pretrained weights
        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, path):
        try:
            state_dict = torch.load(path, map_location='cpu')
            # Extract standard parameter dict if wrapped
            if 'params' in state_dict:
                state_dict = state_dict['params']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            self.backbone.load_state_dict(state_dict, strict=True)
            print(f"Successfully loaded pretrained backbone from {path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights from {path}. Error: {e}")

    def forward(self, x):
        # x is [B, 1, 128, 128]
        
        # Grayscale Adapter: project 1 channel to 3 channels for backbone
        x_3c = x.repeat(1, 3, 1, 1)
        
        # Extract features (restored feature rep)
        features = self.backbone(x_3c)
        
        # Upsampling module with PixelShuffle: [B, 3, 128, 128] -> [B, 4, 128, 128] -> [B, 1, 256, 256]
        out = self.upsampler(features)

        # Residual SR skip gives the network a strong interpolation baseline and
        # lets the trainable path focus on denoising and high-frequency detail.
        skip = torch.nn.functional.interpolate(
            x,
            scale_factor=self.upscale,
            mode="bilinear",
            align_corners=False,
        )
        out = out + skip
        
        return out
