import time
import numpy as np
import tensorflow as tf


@tf.function
def train_step(model, optimizer, x, y, z, clip_norm=1.0):
    """Training step with gradient clipping"""
    with tf.GradientTape() as tape:
        logits = model([x, y, z], training=True)
        loss_value = model.losses

    grads = tape.gradient(loss_value, model.trainable_weights)

    # Gradient clipping to prevent exploding gradients
    grads, _ = tf.clip_by_global_norm(grads, clip_norm)

    # Check for NaN gradients
    has_nan = tf.reduce_any(
        [tf.reduce_any(tf.math.is_nan(g)) for g in grads if g is not None]
    )

    # Only apply gradients if they're valid
    tf.cond(
        has_nan,
        lambda: tf.print("Warning: NaN gradients detected, skipping update"),
        lambda: optimizer.apply_gradients(zip(grads, model.trainable_weights)),
    )

    return loss_value, has_nan


@tf.function
def test_step(model, x, y, z):
    """Validation step"""
    logits = model([x, y, z], training=False)
    loss_value = model.losses
    return loss_value


def get_current_learning_rate(optimizer):
    """Get current learning rate, handling both constant and schedule cases"""
    if hasattr(optimizer.learning_rate, "numpy"):
        # Constant learning rate
        return optimizer.learning_rate.numpy()
    elif hasattr(optimizer.learning_rate, "__call__"):
        # Learning rate schedule - get current value based on optimizer iterations
        return float(optimizer.learning_rate(optimizer.iterations))
    else:
        # Fallback
        return float(optimizer.learning_rate)


def train_loop_stable(
    model,
    optimizer,
    train_data,
    val_data,
    path,
    batch_size,
    num_epochs,
    train_loss_metric,
    val_loss_metric,
    early_stopping_patience=10,
    clip_norm=1.0,
    lr_reduce_patience=5,
):
    """
    Improved training loop with:
    - Gradient clipping
    - NaN detection
    - Early stopping
    - Learning rate reduction
    - Better monitoring
    """
    best_val_loss = np.inf
    patience_counter = 0
    lr_reduce_counter = 0
    initial_lr = get_current_learning_rate(optimizer)

    # History tracking
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        current_lr = get_current_learning_rate(optimizer)
        print(f"Learning rate: {current_lr:.6f}")

        start_time = time.time()
        nan_count = 0

        # Training
        for step, (x, y, z) in enumerate(train_data):
            loss_value, has_nan = train_step(model, optimizer, x, y, z, clip_norm)

            if has_nan:
                nan_count += 1
                print(f"NaN detected at step {step}")
                continue

            train_loss_metric.update_state(loss_value)

            if step % 50 == 0:
                print(
                    f"Step {step}: loss = {float(train_loss_metric.result().numpy()):.4f}"
                )

        # Get epoch metrics
        train_loss = float(train_loss_metric.result().numpy())

        # Check for training issues
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN gradients detected during training")

        if np.isnan(train_loss) or np.isinf(train_loss):
            print("ERROR: Training loss is NaN or Inf. Stopping training.")
            break

        print(f"Training loss: {train_loss:.4f}")
        history["train_loss"].append(train_loss)
        train_loss_metric.reset_states()

        # Validation
        for x, y, z in val_data:
            val_loss_metric.update_state(test_step(model, x, y, z))

        val_loss = float(val_loss_metric.result().numpy())

        if np.isnan(val_loss) or np.isinf(val_loss):
            print("ERROR: Validation loss is NaN or Inf. Stopping training.")
            break

        print(f"Validation loss: {val_loss:.4f}")
        history["val_loss"].append(val_loss)
        val_loss_metric.reset_states()

        # Model checkpointing
        if val_loss < best_val_loss:
            print(
                f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}"
            )
            best_val_loss = val_loss
            model.save_weights(filepath=path, overwrite=True, save_format="h5")
            patience_counter = 0
            lr_reduce_counter = 0
        else:
            patience_counter += 1
            lr_reduce_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}"
            )

            # Reduce learning rate if needed
            if lr_reduce_counter >= lr_reduce_patience:
                # For schedules, we can't directly modify the learning rate
                # So we'll skip LR reduction if using a schedule
                if hasattr(optimizer.learning_rate, "numpy"):
                    # Only reduce if using constant learning rate
                    new_lr = optimizer.learning_rate * 0.5
                    optimizer.learning_rate.assign(new_lr)
                    print(f"Reducing learning rate to {new_lr.numpy():.6f}")
                    lr_reduce_counter = 0
                else:
                    print("Learning rate reduction skipped (using schedule)")
                    lr_reduce_counter = 0

        history["lr"].append(get_current_learning_rate(optimizer))

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        print(f"Time taken: {time.time() - start_time:.2f}s")

    return history


def compute_loss_stable(model, data):
    """Compute loss with NaN checking"""
    losses = []
    for step, (x, y, z) in enumerate(data):
        loss = test_step(model, x, y, z)[0].numpy()
        if np.isnan(loss) or np.isinf(loss):
            print(f"Warning: NaN/Inf loss detected at step {step}, skipping")
            continue
        losses.append(loss)

    if len(losses) == 0:
        print("ERROR: All losses were NaN/Inf")
        return np.array([np.inf])

    return np.array(losses)
