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


def create_lr_scheduler(
    initial_lr=2e-5, warmup_proportion=0.1, decay_steps=None, end_lr=1e-6, power=1.0
):
    """
    Create a learning rate scheduler with warmup and polynomial decay.

    Args:
        initial_lr: Initial learning rate
        warmup_proportion: Proportion of training to perform warmup for
        decay_steps: Number of steps for the decay (usually num_train_steps - num_warmup_steps)
        end_lr: Final learning rate after decay
        power: Power factor for polynomial decay

    Returns:
        A learning rate schedule function
    """

    def lr_scheduler(step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(warmup_proportion * decay_steps, tf.float32)

        # Warmup phase
        warmup_lr = initial_lr * (step / warmup_steps)

        # Decay phase
        decay_progress = tf.clip_by_value(
            (step - warmup_steps) / (decay_steps - warmup_steps), 0.0, 1.0
        )
        decay_factor = (1.0 - decay_progress) ** power
        decay_lr = end_lr + (initial_lr - end_lr) * decay_factor

        # Use warmup_lr for warmup phase, decay_lr for decay phase
        lr = tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: decay_lr)

        return lr

    return lr_scheduler


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
    num_epochs=80,
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
        decay_steps = num_train_steps  # Full schedule over all steps
        end_lr = lr * end_lr_factor

        # Create learning rate scheduler
        lr_schedule = create_lr_scheduler(
            initial_lr=lr,
            warmup_proportion=warmup_proportion,
            decay_steps=decay_steps,
            end_lr=end_lr,
            power=1.0,  # Linear decay
        )

        # Create optimizer with LR schedule
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        )

        # Compile the model
        classifier.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[f1_m, tf.keras.metrics.CategoricalAccuracy()],
        )

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
            # Reduce learning rate when training plateaus (backup to scheduler)
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_f1_m",
                factor=0.5,
                patience=patience // 2,  # Half the patience of early stopping
                verbose=1,
                mode="max",
                min_delta=min_delta,
                cooldown=2,
                min_lr=end_lr / 10,
            ),
            # Log training progress
            tf.keras.callbacks.CSVLogger(
                os.path.join(os.path.dirname(path), "training_log.csv"),
                separator=",",
                append=False,
            ),
            # Create a custom callback to print current learning rate
            tf.keras.callbacks.LambdaCallback(
                on_epoch_begin=lambda epoch, logs: print(
                    f"\nEpoch {epoch+1}: Current learning rate: {classifier.optimizer.learning_rate(classifier.optimizer.iterations).numpy():.8f}"
                )
            ),
        ]

        # Train the model
        print(f"\nTraining with {len(x_train)} samples, {num_train_steps} steps")
        print(
            f"Learning rate: initial={lr}, end={end_lr}, warmup_proportion={warmup_proportion}"
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

            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

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
        except:
            print(
                "Could not create training history plot. Make sure matplotlib is installed."
            )

        del classifier

    # Load model with best weights
    classifier = TFAutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    classifier.load_weights(path)
    return classifier
