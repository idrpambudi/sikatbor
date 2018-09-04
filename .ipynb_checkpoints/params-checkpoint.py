from model.vgg16 import VGG16

image_size = (227, 128)
input_shape = image_size + (3,)

max_epochs = 100
batch_size = 4

threshold = 0.5

model_factory = VGG16
