import os
import zipfile
from PIL import Image
import numpy as np 
import tensorflow as tf
from keras import layers,models

zip_path = 'data/archive (9).zip'
extract_dir = 'data/PetImages'

# extract images 
if not os.path.exists(extract_dir):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data/')
    print("Extraction complete.")

# checks if image is corrupted or not 
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join(extract_dir, folder_name)

    if os.path.exists(folder_path):
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)

            try:
                img = Image.open(fpath)
                img.verify()
            except Exception:
                num_skipped += 1
                os.remove(fpath)

print(f"Deleted {num_skipped} corrupted images.")

#training dataset 
train_ds = tf.keras.utils.image_dataset_from_directory(
    'data/PetImages',
    validation_split=0.2, # 80% to train 20% to test 
    subset='training',
    image_size=(180,180), #reshapes every image to 180x180
    batch_size = 32,
    seed = 123  #prevents data leakage,cuts at the same spot if seed is equal 
)

#testing dataset 
test_ds = tf.keras.utils.image_dataset_from_directory(
    'data/PetImages',
    validation_split =0.2,
    subset = 'validation',
    image_size = (180,180),
    batch_size = 32,
    seed = 123
)

# The optimizer
AUTOTUNE = tf.data.AUTOTUNE #maximizes your training speed without crashing your computer instead of (buffer=32) 
#xit adjusts value based on CPU GPU RAM performance
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)# shuffle:it fills the first 1000 images then it gives 32 randomized images
test_ds = test_ds.cache().prefetch(AUTOTUNE)


#Because the flips, zooms, and rotations are random, the model almost never sees the exact same image twice across all 10 epochs
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
])

model = models.Sequential([
    layers.Input(shape=(180,180,3)),
    data_augmentation,
    layers.Rescaling(1./255),

    layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'),
    layers.MaxPool2D(pool_size=(2,2)),

    layers.Conv2D(filters=64 ,kernel_size=(3,3) , activation='relu'),
    layers.MaxPool2D(pool_size=(2,2)),

    layers.Conv2D(filters=128 ,kernel_size=(3,3) , activation='relu'),
    layers.MaxPool2D(pool_size=(2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1,activation='sigmoid')
])
 
#set rules 
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']    
)
model.summary()
    
history = model.fit(train_ds,validation_data=test_ds, epochs=10)

test_loss,test_acc = model.evaluate(test_ds)
print(f"Final accuracy {test_acc*100:.2f} %")
model.save("cats_vs_dogs_model.keras")

print("Model saved successfully.")  