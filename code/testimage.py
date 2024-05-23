import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

def show_image(img, title=None):
    # Convert the tensor to a numpy array and denormalize
    img = img.numpy().transpose((1, 2, 0))
    img = np.clip((img * 0.5) + 0.5, 0, 1)  # Denormalize if needed (mean=0.5, std=0.5)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')

# Load an example image
img_path = '../data/fundus_757/3.jpg'
img = Image.open(img_path).convert('RGB')


# Define normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.224, 0.225])

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# Apply the transform to the image
original_img = transforms.ToTensor()(img)
normalized_img = transform(img)

# Display the original and normalized images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
show_image(original_img, title='Original Image')

plt.subplot(1, 2, 2)
show_image(normalized_img, title='Normalized Image')

plt.show()
