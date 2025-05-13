import os
import numpy as np
import tensorflow as tf
from model.metrics import f1_m
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer


def __finetune_preprocess__(x, y, model_name, batch_size, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    x_tokenized = tokenizer(
        x,
        return_tensors="tf",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    x_tokenized = {i: x_tokenized[i] for i in tokenizer.model_input_names}
    del tokenizer
    return tf.data.Dataset.from_tensor_slices((x_tokenized, y)).batch(batch_size)


def finetune(
    x_train,
    y_train,
    x_validation,
    y_validation,
    max_length,
    num_labels,
    path,
    model_name="bert-base-uncased",
    lr=3e-5,  # Slightly higher initial learning rate
    num_epochs=60,
    batch_size=16,
    first_layers_to_freeze=10,
    patience=10,  # Early stopping patience
    min_delta=0.001,  # Minimum improvement required
    warmup_proportion=0.1,  # Proportion of training for warmup
    end_lr_factor=0.1,  # Final lr = initial_lr * end_lr_factor
    train=True,
    verbose=1,  # Training verbosity (0, 1, or 2)
):
    if train:
        classifier = TFAutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

        train_data = __finetune_preprocess__(
            x_train, y_train, model_name, batch_size, max_length
        )
        dev_data = __finetune_preprocess__(
            x_validation, y_validation, model_name, batch_size, max_length
        )

        # Freeze specified layers
        for i in range(first_layers_to_freeze):
            classifier.bert.encoder.layer[i].trainable = False

        # Calculate learning rate schedule parameters
        num_train_steps = (len(x_train) // batch_size) * num_epochs

        # Use TF 2.8 compatible learning rate schedule
        warmup_steps = int(warmup_proportion * num_train_steps)

        # Create learning rate schedule - TF 2.8 compatible
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=lr,
            decay_steps=num_train_steps - warmup_steps,
            end_learning_rate=lr * end_lr_factor,
            power=1.0,
        )

        # Create a wrapper function for the learning rate schedule with warmup
        def lr_with_warmup(step):
            step = tf.cast(step, tf.float32)
            if step < warmup_steps:
                # Linear warmup
                warmup_pct = step / tf.cast(warmup_steps, tf.float32)
                return tf.cast(warmup_pct * lr, tf.float32)
            else:
                # Use polynomial decay after warmup
                return lr_schedule(step - warmup_steps)

        # Create optimizer with custom learning rate function
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_with_warmup, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        )

        # Compile the model
        classifier.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[f1_m, tf.keras.metrics.CategoricalAccuracy()],
        )

        # Create a custom learning rate logger callback
        class LRLogger(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                # Get current step
                step = epoch * (len(x_train) // batch_size)
                # Calculate current LR using our function
                current_lr = lr_with_warmup(step)
                if isinstance(current_lr, tf.Tensor):
                    current_lr = current_lr.numpy()
                print(f"\nEpoch {epoch+1}: Current learning rate: {current_lr:.8f}")

        # Create directory for the path if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Create callbacks
        callbacks = [
            # Save best model
            tf.keras.callbacks.ModelCheckpoint(
                filepath=path,
                monitor="val_f1_m",
                mode="max",
                save_weights_only=True,
                save_best_only=True,
                verbose=1,
            ),
            # Early stopping when training plateaus
            tf.keras.callbacks.EarlyStopping(
                monitor="val_f1_m",
                min_delta=min_delta,
                patience=patience,
                verbose=1,
                mode="max",
                restore_best_weights=True,
            ),
            # Reduce learning rate when training plateaus (TF 2.8 compatible version)
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_f1_m",
                factor=0.5,
                patience=patience // 2,
                verbose=1,
                mode="max",
                min_delta=min_delta,
                cooldown=2,
                min_lr=lr * end_lr_factor / 10,
            ),
            # Log training progress
            tf.keras.callbacks.CSVLogger(
                os.path.join(os.path.dirname(path), "training_log.csv"),
                separator=",",
                append=False,
            ),
            # Custom LR logger
            LRLogger(),
        ]

        # Train the model
        print(f"\nTraining with {len(x_train)} samples, {num_train_steps} steps")
        print(
            f"Learning rate: initial={lr}, end={lr * end_lr_factor}, warmup_proportion={warmup_proportion}"
        )
        print(f"Early stopping patience: {patience}, min_delta: {min_delta}")

        history = classifier.fit(
            train_data,
            validation_data=dev_data,
            epochs=num_epochs,
            callbacks=callbacks,
            verbose=verbose,
        )

        # Print training summary
        best_epoch = np.argmax(history.history["val_f1_m"]) + 1
        best_val_f1 = np.max(history.history["val_f1_m"])
        print(
            f"\nTraining completed. Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}"
        )

        # Plot training history if matplotlib is available
        try:
            import matplotlib.pyplot as plt

            # Plot training & validation F1 score
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(history.history["f1_m"])
            plt.plot(history.history["val_f1_m"])
            plt.axvline(x=best_epoch - 1, color="r", linestyle="--")
            plt.title(f"Model F1 Score (Best: {best_val_f1:.4f} at epoch {best_epoch})")
            plt.ylabel("F1 Score")
            plt.xlabel("Epoch")
            plt.legend(["Train", "Validation", "Best Epoch"], loc="lower right")

            # Plot training & validation loss
            plt.subplot(1, 2, 2)
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.axvline(x=best_epoch - 1, color="r", linestyle="--")
            plt.title("Model Loss")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend(["Train", "Validation", "Best Epoch"], loc="upper right")

            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(path), "training_history.png"))
            plt.close()

            print(
                f"Training history plot saved to {os.path.join(os.path.dirname(path), 'training_history.png')}"
            )
        except Exception as e:
            print(f"Could not create training history plot: {str(e)}")

        del classifier

    # Load model with best weights
    classifier = TFAutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    classifier.load_weights(path)
    return classifier
