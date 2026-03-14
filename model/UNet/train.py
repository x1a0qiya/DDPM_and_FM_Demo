import os
from model import UNet
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

"""
Training parameters
"""
epochs = 100
batch_size = 32
image_size = 128

rd_seed = 57
torch.manual_seed(rd_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(rd_seed)

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

path = "../../data"
cat_dataset = datasets.ImageFolder(root = path, transform = transform)
print(f"Successfully loaded the data with size {len(cat_dataset)}")

dataloader = DataLoader(cat_dataset, batch_size = batch_size, shuffle = True, drop_last = True)

"""
UNet
"""
unet = UNet(in_channels = 3, out_channels = 3, base_channels = 64).to(device)
optimizer = torch.optim.AdamW(unet.parameters(), lr = 1e-4, weight_decay = 1e-4)

"""
EMA
"""
ema_avg = get_ema_multi_avg_fn(0.999)
ema_unet = AveragedModel(unet, multi_avg_fn=ema_avg)

"""
Training
"""
save_interval = 10
resume_path = f"checkpoints/checkpoint_epoch_[].pth"  # Resume from here.

if os.path.exists(resume_path):
    print(f"Loading checkpoint from {resume_path}...")
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

    unet.load_state_dict(checkpoint['model_state_dict'])
    ema_unet.load_state_dict(checkpoint['ema_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch']
    print(f"Successfully loaded. Resuming from epoch {start_epoch}")
else:
    start_epoch = 0
    print("No checkpoint found, starting from scratch.")

for epoch in range(start_epoch, epochs):
    epoch_loss = 0.0
    unet.train()

    for step, (images, _) in enumerate(dataloader):
        b = images.shape[0]

        x1 = images.to(device)
        x0 = torch.randn_like(x1).to(device)

        t = torch.rand([b, 1, 1, 1], device = device)
        
        xt = t * x1 + (1 - t) * x0
        
        v_target = x1 - x0
        v_pre = unet(xt, t.view(b))

        loss = nn.functional.mse_loss(v_pre, v_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_unet.update_parameters(unet) 

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch + 1} / {epochs}] Average Loss : {avg_loss:.6f}")

    torch.save({
        'model_state_dict': unet.state_dict(),
        'ema_model_state_dict': ema_unet.state_dict(),
    }, "checkpoints/unet_latest.pth")

    if (epoch + 1) % save_interval == 0:
        checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch + 1}.pth"

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': unet.state_dict(),
            'ema_model_state_dict': ema_unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to '{checkpoint_path}'")

print("Traning Complete")