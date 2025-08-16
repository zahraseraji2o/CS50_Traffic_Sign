import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

# تنظیمات کلی
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # بررسی آرگومان‌های ورودی
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # بارگذاری داده‌ها
    images, labels = load_data(sys.argv[1])

    # تقسیم به داده‌های آموزش و تست
    labels = tf.keras.utils.to_categorical(labels, NUM_CATEGORIES)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # ساخت مدل
    model = get_model()

    # آموزش مدل
    model.fit(x_train, y_train, epochs=EPOCHS)

    # ارزیابی مدل
    model.evaluate(x_test, y_test, verbose=2)

    # ذخیره مدل در صورت نیاز
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    بارگذاری تصاویر از فولدر
    انتظار داریم که فولدر شامل زیرفولدرهایی از 0 تا NUM_CATEGORIES-1 باشد
    """

    images = []
    labels = []

    for category in range(NUM_CATEGORIES):
        folder = os.path.join(data_dir, str(category))
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            # خواندن تصویر
            img = cv2.imread(img_path)
            # تغییر سایز به 30x30
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(category)

    return images, labels


def get_model():
    """
    تعریف یک مدل ساده‌ی شبکه عصبی کانولوشنی
    """

    model = tf.keras.Sequential([

        # لایه کانولوشن + مکس پولینگ
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # لایه دوم کانولوشن + مکس پولینگ
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # تخت کردن داده‌ها
        tf.keras.layers.Flatten(),

        # لایه Dense مخفی
        tf.keras.layers.Dense(128, activation="relu"),

        # خروجی
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # کامپایل مدل
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
