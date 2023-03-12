I know most of the material, so I don't take notes extensively and only write the new stuff

# L1 - Gradient Descent and Backprop

- Higher dimension make it less likely to have local minimas
- SGD makes use of the redundancy of information in batches.
  - We only batch examples because the hardware likes it.
- **Batch Size**
  - in Classification with **k** classes, if your batch size is larger than **2k**, then you are wasting computation.

****

# L3 - Parameter sharing

## RNNs

### RNN Tricks

- Clipping gradients (avoid exploding gradients)

- Leaky integration (propagate long-term dependencies)
  - Talked about later

- Momentum (cheap 2nd-order)
- Initialization (start in right ballpark and avoid exploding/vanishing)

- Sparse Gradients (symmetry breaking)

- Gradient propagation regularizer (avoid vanishing gradients)
- LSTM self-loops (avoid vanishing gradient)