import torch
import matplotlib.pyplot as plt
from model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_size = 128
loaded_model = UNet(in_channels=3, out_channels=3, base_channels=64).to(device)

save_path = "checkpoints/unet_latest.pth" 
unet_data = torch.load(save_path, map_location=device, weights_only=True)

if isinstance(unet_data, dict) and 'ema_model_state_dict' in unet_data:
    print("Found EMA weights, preparing to load...")
    ema_state_dict = unet_data['ema_model_state_dict']
    
    clean_state_dict = {}
    for k, v in ema_state_dict.items():
        if k == 'n_averaged':
            continue
            
        if k.startswith('module.'):
            new_key = k[7:]
        else:
            new_key = k
        clean_state_dict[new_key] = v
        
    loaded_model.load_state_dict(clean_state_dict)
    print("Successfully loaded the EMA U-Net model!")

else:
    loaded_model.load_state_dict(unet_data['model_state_dict'])
    print("Warning: EMA weights not found, loaded standard U-Net weights.")

loaded_model.eval()
print("Successfully load the U-Net model, start drawing")

batch_size = 1
num_steps = 50  
dt = 1.0 / num_steps

with torch.no_grad():
    x_gen = torch.randn(batch_size, 3, image_size, image_size).to(device)
    
    for i in range(num_steps):
        t = i * dt
        
        t_tensor = torch.full((batch_size,), t, device=device)

        v_pred = loaded_model(x_gen, t_tensor)
        
        x_gen = x_gen + v_pred * dt

img_tensor = x_gen.cpu()
img_display = (img_tensor + 1.0) / 2.0
img_display = torch.clamp(img_display, 0.0, 1.0)
img_np = img_display[0].permute(1, 2, 0).numpy()

plt.figure(figsize=(6, 6))
plt.imshow(img_np)
plt.axis('off')
plt.title(f"Generated Cat via Flow Matching ({num_steps} steps)")
plt.show()