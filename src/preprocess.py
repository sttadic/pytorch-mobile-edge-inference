from PIL import Image
from torchvision import transforms

# Standard ImageNet preprocessing
preprocess = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
  )
])


def load_image(path):
  with Image.open(path).convert("RGB") as img:
    tensor = preprocess(img)
  tensor = tensor.unsqueeze(0)  # add batch dimension
  return tensor
