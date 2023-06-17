# Import necessary libraries
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

# Set the path to the pre-trained model
model_path = 'models/RRDB_ESRGAN_x4.pth'

# Set the device to be used (cuda if available, else cpu)
device = torch.device('cuda')
# device = torch.device('cpu')

# Set the path to the folder containing the test images
test_img_folder = 'LR/*'

# Load the pre-trained model and move it to the device
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

# Print the model path and start testing
print('Model path {:s}. \nTesting...'.format(model_path))

# Loop through all images in the test folder
idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # Normalize the pixel values to [0, 1]
    img = img * 1.0 / 255
    # Convert the image to a PyTorch tensor and move it to the device
    img = torch.from_numpy(np.transpose(
        img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

# Perform the super-resolution using the pre-trained model
    with torch.no_grad():
        output = model(img_LR).data.squeeze(
        ).float().cpu().clamp_(0, 1).numpy()

    # Convert the output image to the original color space and scale it to [0, 255]
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    # Save the output image
    cv2.imwrite('results/{:s}_rlt.png'.format(base), output)

    cv2.imwrite('iresults/{:s}_rlt_optimised.jpeg'.format(base),
                output, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
