import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
import scipy.spatial
import sys
sys.path.append('../Python_libs')
import NNlibs
import torch

def generate_convex_shape(image_size=256):
    """Generate a binary image with a convex polygon shape."""
    img = np.ones((image_size, image_size), dtype=np.uint8)   # White background

    # Generate random convex hull points
    num_vertices = np.random.randint(5, 10)  # Number of vertices (5-10 for variety)
    points = np.random.randint(10, image_size - 10, size=(num_vertices, 2))
    
    # Compute convex hull
    hull = scipy.spatial.ConvexHull(points)
    hull_vertices = points[hull.vertices]

    # Create a filled polygon (shape is black)
    rr, cc = polygon(hull_vertices[:, 0], hull_vertices[:, 1], img.shape)
    img[rr, cc] = 0  # Shape is black

    return img

# Generate a dataset of convex shapes
num_samples = 1000  # Number of images
dataset = [generate_convex_shape() for _ in range(num_samples)]

# Display some sample images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for ax, img in zip(axes.flat, dataset[20:30]):
    ax.imshow(img, cmap='gray')
    ax.axis('off')
#plt.show()
plt.savefig('sample_images.png',bbox_inches='tight')

model = NNlibs.CVAE()

#print(np.shape(dataset[1].reshape(1,1,32,32)))
sample_input = torch.randn(8, 1, 32, 32)
#sample = torch.tensor(dataset[1].reshape(1,1,32,32),dtype=torch.float32)

print(np.shape(dataset))

#print(model(sample))
#print(model(sample_input))
