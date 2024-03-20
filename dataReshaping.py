import cv2
import os

def resize_images(input_dir, target_size=(756, 1008)):
    for i in range(1, 11):
        n = str(i)
        img_path = os.path.join(input_dir, n + '.JPG')
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, target_size)
        cv2.imwrite(img_path, resized_img)

def main():
    input_dir = 'data/train/50'
    resize_images(input_dir)

if __name__ == "__main__":
    main()
