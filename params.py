from model.vgg16 import VGG16

folder_train = 'input/train/'
folder_val = 'input/valid/'
folder_test = 'input/test/'
labels_dir = 'input/labels.csv'

image_size = (242, 128)
input_shape = image_size + (3,)

max_epochs = 250
batch_size = 4

threshold = 0.5

model_factory = VGG16
