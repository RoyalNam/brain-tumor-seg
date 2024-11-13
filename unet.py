import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Contracting Path (Encoder)
        self.enc_c1 = self.contracting_block(in_channels, 32)
        self.enc_c2 = self.contracting_block(32, 64)
        self.enc_c3 = self.contracting_block(64, 128)
        self.enc_c4 = self.contracting_block(128, 256)
        self.enc_c5 = self.contracting_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)
        
        # Expanding Path (Decoder)
        self.up_c6 = self.expanding_block(1024, 512)
        self.up_c7 = self.expanding_block(512, 256)
        self.up_c8 = self.expanding_block(256, 128)
        self.up_c9 = self.expanding_block(128, 64)
        self.up_c10 = self.expanding_block(64, 32)
        
        # Final Output Layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def contracting_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
    def expanding_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Contracting path
        e1 = self.enc_c1(x)
        e2 = self.enc_c2(e1)
        e3 = self.enc_c3(e2)
        e4 = self.enc_c4(e3)
        e5 = self.enc_c5(e4)
        
        # Bottleneck
        bottleneck = self.bottleneck(e5)
        
        # Expanding path (with skip connections)
        d6 = self.up_c6(bottleneck)
        d6 = torch.cat((d6, e5), dim=1)
        d7 = self.up_c7(d6)
        d7 = torch.cat((d7, e4), dim=1)
        d8 = self.up_c8(d7)
        d8 = torch.cat((d8, e3), dim=1)
        d9 = self.up_c9(d8)
        d9 = torch.cat((d9, e2), dim=1)
        d10 = self.up_c10(d9)
        d10 = torch.cat((d9, e1), dim=1)
        
        # Final output
        final = self.final_conv(d10)
        return final

if __name__=='__main__':
    model = UNet()
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params}')