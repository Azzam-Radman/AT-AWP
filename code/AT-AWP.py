# imports
import gc
import copy
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow.keras import layers

original_lr = 1e-3
nb_iters = 1
pdg_step_size = 0.3
pgd_steps = [1]*3 + [2]*6
awp_steps = 1
awp_step_size = 0.3
EPS = 1E-20

# Detect hardware
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
except ValueError:
    tpu = None
    gpus = tf.config.experimental.list_logical_devices("GPU")
    
# Select appropriate distribution strategy
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu) # Going back and forth between TPU and host is expensive. Better to run 128 batches on the TPU before reporting back.
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])  
elif len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
elif len(gpus) == 1:
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    print('Running on single GPU ', gpus[0].name)
else:
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    print('Running on CPU')
print("Number of accelerators: ", strategy.num_replicas_in_sync)

loss_object = tf.keras.losses.MeanSquaredError()
awp_optimizer = tf.keras.optimizers.SGD(learning_rate=original_lr/nb_iters)

# Example Perturbation
def clip_grads_by_example_norms(grads, examples, pdg_step_size=1):
    norms = tf.cast(tf.norm(examples, ord='fro', axis=(0, 1), keepdims=True), tf.float32)
    grads_norms = tf.cast(tf.norm(grads, ord='fro', axis=(0, 1), keepdims=True), tf.float32)
    normalized_grads = grads / (grads_norms + EPS)
    clipped_grads = pdg_step_size * normalized_grads * norms
    return clipped_grads

def generate_adv_examples(x, y, pgd_steps=1):
    original_x = tf.cast(copy.deepcopy(x), tf.float32)
    x = tf.cast(x, tf.float32)
    
    for idx in range(pgd_steps):
        with tf.GradientTape() as tape:
            tape.watch(x)
            pred = model(x, training=True)
            loss = - loss_object(y, pred)
            
        grads = tape.gradient(loss, x)
        clipped_grads = clip_grads_by_example_norms(grads, original_x, pdg_step_size=pdg_step_size)
        x -= clipped_grads
        
    return x
  
# Weight Perturbation
def normalize(perturbations, weights, step_size=awp_step_size):
    weights_norm = tf.norm(weights, ord='fro', axis=(0, 1))
    perturbations_norm = tf.norm(perturbations, ord='fro', axis=(0, 1))

    return perturbations * (weights_norm / (perturbations_norm + EPS))
  
def normalize_grad_by_weights(grads, ref_weights, awp_step_size=awp_step_size):
    normalized_grads = copy.deepcopy(grads)
    for i in range(len(ref_weights)):
        if len(ref_weights[i].shape) <= 1:
            normalized_grads[i] *= 0    # ignore perturbations with 1 dimension (e.g. BN, bias)
        else:
            normalized_grads[i] = normalize(grads[i], tf.constant(ref_weights[i]), step_size=awp_step_size)
    return normalized_grads
  
# Adversarial Weight Perturbation¶
def awp(adv_x, y, awp_steps=1, awp_step_size=awp_step_size):
    old_w = model.trainable_weights

    for idx in range(awp_steps):
        with tf.GradientTape() as tape:
            pred = model(adv_x, training=True)
            loss = - loss_object(y, pred)  # multiply by -1 to get the gradients in the direction that maximizes loss

        grads = tape.gradient(loss, model.trainable_weights)
        perturbations = normalize_grad_by_weights(grads, old_w, awp_step_size=awp_step_size)
        # update wieghts using perturbations
        awp_optimizer.apply_gradients(zip(perturbations, model.trainable_weights))

    diff = [w1 - w2 for w1, w2 in zip(model.trainable_weights, old_w)]
    
    return diff
  
# Adversarial Training Step with Perturbed Weights and Examples
def train_step(optimizer):
    @tf.function
    def one_train_step(adv_x, y, diff, is_adv):
        with tf.GradientTape() as tape:
            pred = model(adv_x, training=True)
            loss = loss_object(y, pred)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if is_adv:
            # subtract differences from the original weights
            restored_weights = [w1 - w2 for w1, w2 in zip(model.trainable_weights, diff)]
            # assign the subtracted weights to the model
            [model.trainable_weights[i].assign(restored_weights[i]) for i in range(len(restored_weights))]

        return loss
    return one_train_step

# Adversarial training
def do_training():
  optimizer = tf.keras.optimizers.Adam(learning_rate=original_lr)
  training_step = train_step(optimizer)
  for epoch in range(num_epochs):

      print(f'Epoch: {epoch+1}/{num_epochs}', '-'*30 + '>')

      total_losses = []
      # strat adversarial training from the second epoch
      if epoch == 0:
          print('Starting Adversarial Training')
          print('-----------------------------')
      for batch, (x1, x2) in tqdm(enumerate(train_ds), total=len(train_ds)):
          # perturb inputs
          adv_x = generate_adv_examples(x1, x2, pgd_steps=pgd_steps[epoch-1])
          # perturb weights
          diff = awp(adv_x, x2, awp_steps=1)
          # train on perturbed inputs and weights
          total_losses.append(training_step(adv_x, x2, diff=diff, is_adv=True))
          del adv_x, diff
          gc.collect()
      print(f'Mean adversarial loss: {np.mean(total_losses):>7f}')

      # save weights after each epoch
      model.save_weights('weights.h5')
    
   
if __name__ == "main":
  do_training()
  
