from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def create_datagenerator():
    dataGenerator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode="nearest",
    )
    return dataGenerator


def augment_images(img_path, save_dir, prefix, num_images):
    dataGenerator = create_datagenerator()
    img = load_img(img_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in dataGenerator.flow(
        x,
        batch_size=1,
        save_to_dir=save_dir,
        save_prefix=prefix,
        save_format="jpg",
    ):
        i += 1
        if i >= num_images:
            break


def main():
    prefix = "50_"
    num_images_per_file = 20
    for p in range(1, 11):
        num = str(p)
        img_path = "data/train/50/" + num + ".JPG"
        save_dir = "data/train/data50Augmentation"
        augment_images(img_path, save_dir, prefix + num, num_images_per_file)


if __name__ == "__main__":
    main()
