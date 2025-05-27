import os
import numpy as np
import tensorflow as tf
from model.metrics import f1_m
from tensorflow.keras.utils import Sequence
from sklearn.utils.class_weight import compute_class_weight
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


class BalancedDataGenerator(Sequence):
    """Balanced batch generator for imbalanced datasets"""

    def __init__(
        self,
        x,
        y,
        tokenizer,
        max_length,
        batch_size,
        model_name,
        balance_method="undersample",
        random_state=42,
    ):
        self.x = x
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_name = model_name
        self.balance_method = balance_method
        self.random_state = random_state

        # Get unique classes and their counts
        self.unique_classes, self.class_counts = np.unique(
            np.argmax(y, axis=1), return_counts=True
        )
        self.num_classes = len(self.unique_classes)

        # Create class indices mapping
        self.class_indices = {}
        for i, class_id in enumerate(self.unique_classes):
            self.class_indices[class_id] = np.where(np.argmax(y, axis=1) == class_id)[0]

        # Determine samples per class for balanced batching
        if balance_method == "oversample":
            self.samples_per_class = max(self.class_counts)
        elif balance_method == "undersample":
            self.samples_per_class = min(self.class_counts)
        else:  # 'median'
            self.samples_per_class = int(np.median(self.class_counts))

        # Calculate batches per epoch
        self.samples_per_batch_per_class = max(1, self.batch_size // self.num_classes)
        self.batches_per_epoch = (
            self.samples_per_class // self.samples_per_batch_per_class
        )

        np.random.seed(self.random_state)

        # Generate the oversampled dataset once
        self._generate_oversampled_data()

    def _generate_oversampled_data(self):
        """Generate the complete oversampled dataset"""
        self.oversampled_x = []
        self.oversampled_y = []

        np.random.seed(self.random_state)

        for class_id in self.unique_classes:
            class_indices = self.class_indices[class_id]

            # Sample with replacement to reach target size
            if len(class_indices) < self.samples_per_class:
                # Oversample this class
                sampled_indices = np.random.choice(
                    class_indices, size=self.samples_per_class, replace=True
                )
            else:
                # Undersample this class (or keep as is for median)
                sampled_indices = np.random.choice(
                    class_indices, size=self.samples_per_class, replace=False
                )

            # Add samples to oversampled dataset
            for idx in sampled_indices:
                self.oversampled_x.append(self.x[idx])
                self.oversampled_y.append(self.y[idx])

        # Shuffle the oversampled data
        combined = list(zip(self.oversampled_x, self.oversampled_y))
        np.random.shuffle(combined)
        self.oversampled_x, self.oversampled_y = zip(*combined)
        self.oversampled_x = list(self.oversampled_x)
        self.oversampled_y = list(self.oversampled_y)

    def get_oversampled_data(self):
        """
        Returns the complete oversampled dataset as lists

        Returns:
        --------
        oversampled_sentences : list
            List of oversampled sentences
        oversampled_intents : list
            List of oversampled intent labels (one-hot encoded)
        """
        return self.oversampled_x.copy(), self.oversampled_y.copy()

    def get_oversampled_data_flat(self):
        """
        Returns oversampled data with flattened intent labels

        Returns:
        --------
        oversampled_sentences : list
            List of oversampled sentences
        oversampled_intents_flat : list
            List of intent class indices (not one-hot encoded)
        """
        oversampled_intents_flat = [np.argmax(y) for y in self.oversampled_y]
        return self.oversampled_x.copy(), oversampled_intents_flat

    def get_oversampled_stats(self):
        """
        Get statistics about the oversampled dataset

        Returns:
        --------
        dict : Statistics including class distribution
        """
        oversampled_classes = [np.argmax(y) for y in self.oversampled_y]
        class_counts = np.bincount(oversampled_classes)

        return {
            "total_samples": len(self.oversampled_x),
            "samples_per_class": self.samples_per_class,
            "class_distribution": class_counts,
            "balance_method": self.balance_method,
            "original_total": len(self.x),
            "original_class_counts": self.class_counts,
        }

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []

        for class_id in self.unique_classes:
            # Sample indices for this class
            class_idx = np.random.choice(
                self.class_indices[class_id],
                size=self.samples_per_batch_per_class,
                replace=True,
            )

            for i in class_idx:
                batch_x.append(self.x[i])
                batch_y.append(self.y[i])

        # Shuffle the batch
        indices = np.random.permutation(len(batch_x))
        batch_x = [batch_x[i] for i in indices]
        batch_y = [batch_y[i] for i in indices]

        # Tokenize batch
        batch_x_tokenized = self.tokenizer(
            batch_x,
            return_tensors="tf",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Convert to proper format
        batch_features = {
            key: batch_x_tokenized[key] for key in self.tokenizer.model_input_names
        }
        batch_labels = tf.constant(batch_y, dtype=tf.float32)

        return batch_features, batch_labels

    def on_epoch_end(self):
        """Shuffle data at the end of each epoch"""
        np.random.seed(self.random_state + self.batches_per_epoch)


class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss implementation for handling class imbalance"""

    def __init__(self, alpha=1.0, gamma=2.0, from_logits=True, name="focal_loss"):
        super(FocalLoss, self).__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        # Convert to probabilities if logits
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        # Clip predictions to prevent log(0)
        epsilon = 1e-8
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate focal weight
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1 - p_t, self.gamma)

        # Apply alpha weighting if specified
        if isinstance(self.alpha, (list, np.ndarray)):
            alpha_t = tf.reduce_sum(y_true * self.alpha, axis=-1, keepdims=True)
        else:
            alpha_t = self.alpha

        # Calculate focal loss
        focal_loss = alpha_t * focal_weight * cross_entropy

        return tf.reduce_sum(focal_loss, axis=-1)


class WarmupLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with warmup that's compatible with TF 2.8.2"""

    def __init__(
        self,
        warmup_steps,
        initial_lr,
        target_lr,
        decay_schedule=None,
        warmup_strategy="linear",
        name=None,
    ):
        super(WarmupLearningRateSchedule, self).__init__()
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.decay_schedule = decay_schedule
        self.warmup_strategy = warmup_strategy
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmupLearningRateSchedule"):
            step = tf.cast(step, tf.float32)
            warmup_steps = tf.cast(self.warmup_steps, tf.float32)

            # Warmup phase
            if self.warmup_strategy == "linear":
                warmup_lr = self.initial_lr + (self.target_lr - self.initial_lr) * (
                    step / warmup_steps
                )
            elif self.warmup_strategy == "cosine":
                warmup_lr = (
                    self.initial_lr
                    + (self.target_lr - self.initial_lr)
                    * (1 - tf.cos(tf.constant(np.pi) * step / warmup_steps))
                    / 2
                )
            else:  # constant
                warmup_lr = self.initial_lr

            # Post-warmup phase
            if self.decay_schedule is not None:
                # Adjust step for decay schedule (subtract warmup steps)
                decay_step = tf.maximum(0.0, step - warmup_steps)
                decay_lr = self.decay_schedule(decay_step)
            else:
                decay_lr = self.target_lr

            # Return warmup LR if in warmup phase, otherwise decay LR
            return tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: decay_lr)

    def get_config(self):
        return {
            "warmup_steps": self.warmup_steps,
            "initial_lr": self.initial_lr,
            "target_lr": self.target_lr,
            "warmup_strategy": self.warmup_strategy,
            "name": self.name,
        }


def compute_class_weights_dict(y_train, method="balanced"):
    """Compute class weights for imbalanced dataset"""
    y_integers = np.argmax(y_train, axis=1)

    if method == "balanced":
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_integers), y=y_integers
        )
    elif method == "effective_number":
        # Effective Number of Samples method
        samples_per_cls = np.bincount(y_integers)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        class_weights = weights / np.sum(weights) * len(weights)
    else:
        # Inverse frequency
        samples_per_cls = np.bincount(y_integers)
        class_weights = 1.0 / samples_per_cls
        class_weights = class_weights / np.sum(class_weights) * len(class_weights)

    return {i: class_weights[i] for i in range(len(class_weights))}


def create_lr_schedule_with_decay(
    initial_lr,
    decay_steps,
    decay_rate=0.96,
    staircase=True,
    schedule_type="exponential",
):
    """Create learning rate decay schedule"""
    if schedule_type == "exponential":
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
        )
    elif schedule_type == "cosine":
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr, decay_steps=decay_steps
        )
    elif schedule_type == "polynomial":
        return tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            end_learning_rate=initial_lr * 0.01,
        )
    else:
        return None


def finetune(
    x_train,
    y_train,
    x_validation,
    y_validation,
    max_length,
    num_labels,
    path,
    model_name="bert-base-uncased",
    lr=2e-5,
    num_epochs=60,
    batch_size=16,
    first_layers_to_freeze=10,
    train=True,
    # New parameters for imbalanced dataset handling
    use_class_weights=False,
    class_weight_method="balanced",
    use_focal_loss=False,
    focal_alpha=1.0,
    focal_gamma=2.0,
    use_balanced_sampling=False,
    sampling_method="undersample",
    use_warmup=False,
    warmup_steps=None,
    warmup_initial_lr=1e-6,
    warmup_strategy="linear",
    use_lr_schedule=False,
    lr_schedule_type="exponential",
    lr_decay_steps=None,
    lr_decay_rate=0.96,
    random_state=42,
):
    """
    Enhanced finetune function with support for imbalanced datasets

    Parameters:
    -----------
    x_train, y_train: Training data and labels
    x_validation, y_validation: Validation data and labels
    max_length: Maximum sequence length
    num_labels: Number of classes
    path: Path to save model weights
    model_name: Pre-trained model name
    lr: Learning rate
    num_epochs: Number of training epochs
    batch_size: Batch size
    first_layers_to_freeze: Number of layers to freeze
    train: Whether to train the model

    Imbalanced dataset parameters:
    use_class_weights: Use class weights to handle imbalance
    class_weight_method: Method to compute class weights ('balanced', 'effective_number', 'inverse_frequency')
    use_focal_loss: Use focal loss instead of categorical crossentropy
    focal_alpha: Alpha parameter for focal loss
    focal_gamma: Gamma parameter for focal loss
    use_balanced_sampling: Use balanced batch sampling
    sampling_method: Sampling strategy ('oversample', 'undersample', 'median')
    use_warmup: Use learning rate warmup
    warmup_steps: Number of warmup steps
    warmup_initial_lr: Initial learning rate for warmup
    warmup_strategy: Warmup strategy ('linear', 'cosine', 'constant')
    use_lr_schedule: Use learning rate decay schedule
    lr_schedule_type: Type of LR schedule ('exponential', 'cosine', 'polynomial')
    lr_decay_steps: Steps for learning rate decay
    lr_decay_rate: Decay rate for learning rate
    random_state: Random seed
    """

    if train:
        # Create classifier model
        classifier = TFAutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

        # Freeze specified layers
        for i in range(first_layers_to_freeze):
            classifier.bert.encoder.layer[i].trainable = False

        # Prepare loss function
        if use_focal_loss:
            # Compute alpha weights for focal loss if using class weights
            if use_class_weights:
                class_weights = compute_class_weights_dict(y_train, class_weight_method)
                focal_alpha = [class_weights[i] for i in range(num_labels)]
            else:
                focal_alpha = focal_alpha

            loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, from_logits=True)
        else:
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        # Prepare class weights (only if not using focal loss)
        class_weights_dict = None
        if use_class_weights and not use_focal_loss:
            class_weights_dict = compute_class_weights_dict(
                y_train, class_weight_method
            )

        # Prepare learning rate schedule
        lr_schedule = lr
        if use_warmup:
            if warmup_steps is None:
                # Default warmup steps: 10% of total training steps
                steps_per_epoch = len(x_train) // batch_size
                warmup_steps = int(0.1 * num_epochs * steps_per_epoch)

            # Create decay schedule for post-warmup if LR schedule is used
            decay_schedule = None
            if use_lr_schedule and lr_decay_steps is not None:
                decay_schedule = create_lr_schedule_with_decay(
                    initial_lr=lr,
                    decay_steps=lr_decay_steps,
                    decay_rate=lr_decay_rate,
                    schedule_type=lr_schedule_type,
                )

            # Create warmup schedule
            lr_schedule = WarmupLearningRateSchedule(
                warmup_steps=warmup_steps,
                initial_lr=warmup_initial_lr,
                target_lr=lr,
                decay_schedule=decay_schedule,
                warmup_strategy=warmup_strategy,
            )
        elif use_lr_schedule and lr_decay_steps is not None:
            lr_schedule = create_lr_schedule_with_decay(
                initial_lr=lr,
                decay_steps=lr_decay_steps,
                decay_rate=lr_decay_rate,
                schedule_type=lr_schedule_type,
            )

        # Prepare optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Compile model
        classifier.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[f1_m, tf.keras.metrics.CategoricalAccuracy()],
        )

        # Prepare callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(path, "best_model"),
                monitor="val_f1_m",
                mode="max",
                save_weights_only=True,
                save_best_only=True,
            )
        ]

        # Prepare data
        if use_balanced_sampling:
            # Use balanced batch generator
            train_generator = BalancedDataGenerator(
                x=x_train,
                y=y_train,
                tokenizer=AutoTokenizer.from_pretrained(model_name),
                max_length=max_length,
                batch_size=batch_size,
                model_name=model_name,
                balance_method=sampling_method,
                random_state=random_state,
            )

            # Validation data (normal preprocessing)
            dev_data = __finetune_preprocess__(
                x_validation, y_validation, model_name, 1, max_length
            )

            # Train with generator
            classifier.fit(
                train_generator,
                validation_data=dev_data,
                epochs=num_epochs,
                callbacks=callbacks,
                verbose=1,
            )
        else:
            # Use normal data preprocessing
            train_data = __finetune_preprocess__(
                x_train, y_train, model_name, batch_size, max_length
            )
            dev_data = __finetune_preprocess__(
                x_validation, y_validation, model_name, 1, max_length
            )

            # Train with class weights if specified
            classifier.fit(
                train_data,
                validation_data=dev_data,
                epochs=num_epochs,
                callbacks=callbacks,
                class_weight=class_weights_dict,
                verbose=1,
            )

        del classifier

    # Load the best model
    classifier = TFAutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    classifier.load_weights(os.path.join(path, "best_model"))

    return classifier
