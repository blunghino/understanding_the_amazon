from PIL import Image
from numpy.random import rand

from torch import np

# def generate_rotations(image):
#     out = []
#     im1 = image.rotate(90)
#     im2 = image.rotate(180)
#     im3 = image.rotate(270)

#     out.append(im1)
#     out.append(im2)
#     out.append(im3)
#     out.append(image.transpose(Image.FLIP_LEFT_RIGHT))
#     out.append(im1.transpose(Image.FLIP_LEFT_RIGHT))
#     out.append(im2.transpose(Image.FLIP_LEFT_RIGHT))
#     out.append(im3.transpose(Image.FLIP_LEFT_RIGHT))

#     return out


def random_flip_rotation_pil(PIL_image):
    ## Returns a random transformation (flip or rotation) of the image
    rando = rand()
    if rando < 0.25:
        out_image = PIL_image
    elif rando < 0.5:
        out_image = PIL_image.rotate(90)
    elif rando < 0.75:
        out_image = PIL_image.rotate(180)
    else:
        out_image = PIL_image.rotate(270)

    rando_2 = rand()
    if rando_2 < 0.5:
        out_image.transpose(Image.FLIP_LEFT_RIGHT)

    return out_image

def random_flip_rotation_np(a):
    ## Returns a random transformation (flip or rotation) of the image
    rando = rand()
    if rando < 0.25:
        out_image = a
    elif rando < 0.5:
        out_image = np.rot90(a, k=1)
    elif rando < 0.75:
        out_image = np.rot90(a, k=2)
    else:
        out_image = np.rot90(a, k=3)

    rando_2 = rand()
    if rando_2 < 0.5:
        out_image = np.fliplr(out_image)

    return out_image

# if __name__ == '__main__':
#     original_image = Image.open("train_1.jpg")
#     out = random_flip_rotation(original_image)
#     out.save('test.jpg')

#     # for i, image in enumerate(generate_rotations(original_image)):
    #     image.save('test' + str(i) + '.jpg')