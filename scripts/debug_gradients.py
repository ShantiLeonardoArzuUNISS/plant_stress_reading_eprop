"""
debug_gradients.py - VERSIONE REALISTICA
"""

import os
import sys

import torch
import torch.nn as nn

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
utils_dir = os.path.join(parent_dir, "utils")
sys.path.insert(0, parent_dir)
sys.path.insert(0, utils_dir)

from neuron_models import CuBaLIF, CuBaRLIF

print("=" * 80)
print("DEBUG: Test Realistico con Loop Temporale")
print("=" * 80)

# Parametri
batch_size = 4
nb_inputs = 6
nb_hidden = 50
nb_outputs = 3
nb_steps = 20  # Simula 20 timesteps

# Crea layer
rec_layer = CuBaRLIF(
    batch_size=batch_size,
    nb_inputs=nb_inputs,
    nb_neurons=nb_hidden,
    alpha=0.9,
    beta=0.95,
    fwd_scale=0.5,
    rec_scale=0.1,
    firing_threshold=1.0,
    device="cpu",
    dtype=torch.float,
    requires_grad=True,
)

ff_layer = CuBaLIF(
    batch_size=batch_size,
    nb_inputs=nb_hidden,
    nb_neurons=nb_outputs,
    alpha=0.9,
    beta=0.95,
    fwd_scale=0.5,
    firing_threshold=1.0,
    device="cpu",
    dtype=torch.float,
    requires_grad=True,
)

print(f"\n[TEST] Forward Pass con Loop Temporale ({nb_steps} steps)")

# Input spike train
x = torch.randn(batch_size, nb_steps, nb_inputs)  # (batch, time, features)

# Reset states
rec_layer.reset(batch_size)
ff_layer.reset(batch_size)

# Recording tensors
out_spikes = torch.zeros((batch_size, nb_steps, nb_outputs))

# Loop temporale (come nel training vero)
for t in range(nb_steps):
    h_spk, h_syn, h_mem = rec_layer.forward(x[:, t, :])
    o_spk, o_syn, o_mem = ff_layer.forward(h_spk)
    out_spikes[:, t, :] = o_spk

print(f"  Output spikes shape: {out_spikes.shape}")
print(f"  Output spikes requires_grad: {out_spikes.requires_grad}")

# Loss realistica: somma spike su tempo
m = torch.sum(out_spikes, dim=1)  # (batch, nb_outputs)
print(f"  Spike counts: {m.shape}")
print(f"  Spike counts requires_grad: {m.requires_grad}")

# Calcola loss (simulando NLLLoss)
log_softmax = nn.LogSoftmax(dim=1)
log_p_y = log_softmax(m)
target = torch.randint(0, nb_outputs, (batch_size,))  # Label casuali
loss_fn = nn.NLLLoss()
loss = loss_fn(log_p_y, target)

print("\n[TEST] Backward Pass")
print(f"  Loss: {loss.item():.4f}")
print(f"  Loss requires_grad: {loss.requires_grad}")

# Backward
loss.backward()

print("\n[RISULTATO]")
print(f"  ✓ rec_layer.ff_weights.grad: {rec_layer.ff_weights.grad is not None}")
print(f"  ✓ rec_layer.rec_weights.grad: {rec_layer.rec_weights.grad is not None}")
print(f"  ✓ ff_layer.ff_weights.grad: {ff_layer.ff_weights.grad is not None}")

if rec_layer.ff_weights.grad is not None:
    print("\n  Gradient norms:")
    print(f"    rec_ff: {rec_layer.ff_weights.grad.norm().item():.6f}")
    print(f"    rec_rec: {rec_layer.rec_weights.grad.norm().item():.6f}")
    print(f"    ff: {ff_layer.ff_weights.grad.norm().item():.6f}")
    print("\n  🎉 GRADIENTS PROPAGATE CORRECTLY!")
else:
    print("\n  ⚠️  Gradients still not propagating - check computational graph")

print("\n" + "=" * 80)
