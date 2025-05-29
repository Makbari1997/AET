import tensorflow as tf
from tensorflow_probability import distributions as tfd


def vae_cost(y_true, y_pred, mu, log_var, z_sample, analytic_kl=True, kl_weight=1.0):
    """
    Compute VAE loss with numerical stability improvements

    Args:
        y_true: True BERT embeddings
        y_pred: Reconstructed embeddings
        mu: Mean of latent distribution
        log_var: Log variance of latent distribution (for numerical stability)
        z_sample: Sampled latent representation
        analytic_kl: Whether to use analytical KL divergence
        kl_weight: Weight for KL divergence term
    """
    # Reconstruction loss using MSE for continuous embeddings
    reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

    # KL divergence with numerical stability
    if analytic_kl:
        # Using log variance for numerical stability
        # KL(q||p) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_divergence = -0.5 * tf.reduce_sum(
            1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1
        )

        # Add small epsilon to prevent numerical issues
        kl_divergence = tf.maximum(kl_divergence, 1e-6)
    else:
        # Monte Carlo approximation
        log_pz = -0.5 * tf.reduce_sum(tf.square(z_sample), axis=1)
        log_qz_x = -0.5 * tf.reduce_sum(
            log_var + tf.square(z_sample - mu) / tf.exp(log_var), axis=1
        )
        kl_divergence = log_qz_x - log_pz

    # Compute ELBO (Evidence Lower Bound)
    # ELBO = -E[log p(x|z)] + KL(q(z|x)||p(z))
    elbo = tf.reduce_mean(reconstruction_loss + kl_weight * kl_divergence)

    return elbo


class Sampling(tf.keras.layers.Layer):
    """Reparameterization trick with log variance"""

    def call(self, inputs):
        mu, log_var = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.random.normal(shape=(batch, dim))

        # Sample: z = mu + sigma * epsilon, where sigma = exp(0.5 * log_var)
        return mu + tf.exp(0.5 * log_var) * epsilon


def encoder_layers(inputs, latent_dim, dims: list, activation: str):
    """Build encoder layers with proper activations"""
    encoder = inputs

    # Hidden layers
    for dim in dims:
        encoder = tf.keras.layers.Dense(units=dim, activation=activation)(encoder)
        encoder = tf.keras.layers.Dropout(0.2)(encoder)

    # Output layers - no activation for mu and log_var
    mu = tf.keras.layers.Dense(units=latent_dim)(encoder)
    log_var = tf.keras.layers.Dense(units=latent_dim)(encoder)

    # Clip log_var to prevent numerical instability
    log_var = tf.clip_by_value(log_var, -10, 10)

    return mu, log_var, encoder


def encoder_model(input_shape, latent_dim, dims, activation="relu"):
    """Create encoder model"""
    input = tf.keras.layers.Input(shape=input_shape, name="encoder_model_input")
    mu, log_var, encoder = encoder_layers(
        inputs=input, latent_dim=latent_dim, dims=dims, activation=activation
    )
    z = Sampling()((mu, log_var))
    model = tf.keras.Model(inputs=input, outputs=[mu, log_var, z])
    model._name = "Encoder"
    return model


def decoder_layers(inputs, dims, activation):
    """Build decoder layers"""
    dec = inputs

    # Hidden layers
    for dim in dims[:-1]:
        dec = tf.keras.layers.Dense(units=dim, activation=activation)(dec)
        dec = tf.keras.layers.Dropout(0.2)(dec)

    # Output layer - no activation for continuous values
    dec = tf.keras.layers.Dense(units=dims[-1])(dec)

    return dec


def decoder_model(input_shape, dims, activation="relu"):
    """Create decoder model"""
    inputs = tf.keras.layers.Input(shape=input_shape)
    outputs = decoder_layers(inputs, dims, activation)
    model = tf.keras.Model(inputs, outputs)
    model._name = "Decoder"
    return model


def vae(encoder, decoder, bert, input_shape, beta=1.0):
    """
    Create VAE model with improved numerical stability

    Args:
        encoder: Encoder model
        decoder: Decoder model
        bert: BERT model
        input_shape: Input shape
        beta: Beta parameter for beta-VAE (controls KL weight)
    """
    input_ids = tf.keras.layers.Input(
        shape=input_shape, name="input_ids", dtype=tf.int32
    )
    attention_mask = tf.keras.layers.Input(
        shape=input_shape, name="attention_mask", dtype=tf.int32
    )
    token_type_ids = tf.keras.layers.Input(
        shape=input_shape, name="token_type_ids", dtype=tf.int32
    )

    # Get BERT embeddings (CLS token)
    embeddings = bert(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0][:, 0]

    # Encode
    mu, log_var, z = encoder(embeddings)

    # Decode
    reconstructed = decoder(z)

    # Create model
    model = tf.keras.Model(
        inputs=[input_ids, attention_mask, token_type_ids], outputs=reconstructed
    )

    # Add loss with beta parameter
    loss = vae_cost(embeddings, reconstructed, mu, log_var, z, kl_weight=beta)
    model.add_loss(loss)
    model._name = "VAE"

    # Add metrics for monitoring
    model.add_metric(loss, name="total_loss")

    # Separate reconstruction and KL losses for monitoring
    reconstruction_loss = tf.reduce_mean(tf.square(embeddings - reconstructed))
    kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
    model.add_metric(reconstruction_loss, name="reconstruction_loss")
    model.add_metric(kl_loss, name="kl_loss")

    return model
