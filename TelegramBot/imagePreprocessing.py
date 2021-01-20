import torchvision.transforms as transforms

from skimage.feature import hog

new_size = (256,256)
data_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize(new_size)])


def pytorch_preprocessing(img):
    # Resize the image and convert it to tensors
    img = data_transforms(img)

    # Change the image shape from [W, H, C] to [B, C, W, H]
    # Where B = Batch Size, C = Channels (RGB), W = Width, H = Height.
    img = img.unsqueeze(0)

    return img


def sklearn_preprocessing(img):
    img = img.resize(new_size)

    hog_features = hog(img, pixels_per_cell=(20, 20),
                       cells_per_block=(10, 10),
                       orientations=10,
                       block_norm='L2')

    hog_features = hog_features.reshape(1, -1)

    return hog_features
