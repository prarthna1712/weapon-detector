import tensorflow as tf

def load_data(train_dir, val_dir, img_size=(224,224), batch_size=32):
    # Load raw datasets first (with int labels)
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int"
    )

    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int"
    )

    # ✅ Capture class names before mapping
    class_names = raw_train_ds.class_names

    # Normalize images
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = raw_train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds   = raw_val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds, class_names

    def load_test_dataset(test_dir, img_size=(224,224), batch_size=32):
        raw_test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='int',
            shuffle=False
        )

        normalization_layer = tf.keras.layers.Rescaling(1.255)
        test_ds = raw_test_ds.map(lambda x, y:(normalization_layer(x),y))

        return test_ds

