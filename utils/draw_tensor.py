from PIL import Image
from utils.jpegify_tensor import jpegify_tensor
def draw_tensor(x):
    img = Image.fromarray(jpegify_tensor(x).detach().numpy())
    img.show()
    return img