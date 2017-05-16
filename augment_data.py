from PIL import Image

def generate_rotations(image):
    out = []
    im1 = image.rotate(90)
    im2 = image.rotate(180)
    im3 = image.rotate(270)

    out.append(im1)
    out.append(im2)
    out.append(im3)
    out.append(image.transpose(Image.FLIP_LEFT_RIGHT))
    out.append(im1.transpose(Image.FLIP_LEFT_RIGHT))
    out.append(im2.transpose(Image.FLIP_LEFT_RIGHT))
    out.append(im3.transpose(Image.FLIP_LEFT_RIGHT))

    return out


if __name__ == '__main__':
    original_image = Image.open("cat.jpg")
    for i, image in enumerate(transforms(original_image)):
        image.save('test' + str(i) + '.jpg')