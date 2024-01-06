ImageFlow = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255, rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

classes = ['apple', 'banana', 'mixed', 'orange']

for class_name in classes:
    # Create a directory for each class
    image_directory = os.path.join("train", class_name)

    # Iterate over all the files in the directory
    for filename in os.listdir(image_directory):
        if filename.endswith('.jpg'):
            # Open the image file
            image_path = os.path.join(image_directory, filename)
            image = Image.open(image_path)

            # Convert the image to RGB mode
            image = image.convert('RGB')

            # Save the image with the same name
            image.save(image_path)

            # Close the image file
            image.close()

    # Generate augmented images and save them to the train dataset directory
    for filename in os.listdir(image_directory):
        img_path = os.path.join(image_directory, filename)
        img = plt.imread(img_path)  # Read the image using matplotlib
        img = img.reshape((1,) + img.shape)  # Reshape the image for augmentation
        save_dir = image_directory  # Set the directory to save the augmented images
        save_prefix = filename.split('.')[0] # Set the prefix for the saved images
        save_format = 'jpg'  # Set the format for the saved images

        # Generate augmented images and save them
        i = 0
        for batch in ImageFlow.flow(img, save_to_dir=save_dir, save_prefix=save_prefix, save_format=save_format):
            number_of_augmented_images = 6 if class_name == 'mixed' else 3

            i += 1
            if i >= number_of_augmented_images:  # Generate 5 augmented images for each original image
                break