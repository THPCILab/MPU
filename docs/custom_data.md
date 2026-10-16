# Running on your own feature data

The compact single-task driver makes loading and training explicit without
EMNIST or a GPU. It uses continuous operators; finite-codebook multi-task
experiments retain their separate documented protocol.

## Input format

Save a non-pickled NumPy NPZ file:

| Key | Shape and type |
| --- | --- |
| x_train | (N_train, D), finite real numbers |
| y_train | (N_train,), integer labels 0 through C−1 |
| x_test | (N_test, D), finite real numbers |
| y_test | (N_test,), integer labels, no unseen class |

D must be even and at least 2. Every training class must occur.
Fit any centering, normalization or PCA on training data only, then apply the
same transformations to held-out inputs. The driver does not fit PCA or alter
your supplied feature scaling. Do not use held-out labels for model selection.

```python
np.savez("my_features.npz", x_train=x_train.astype("float32"),
         y_train=y_train.astype("int64"), x_test=x_test.astype("float32"),
         y_test=y_test.astype("int64"))
```

## Run

```bash
python examples/train_custom_features.py --input my_features.npz --epochs 20 --sharing-ratio 50 --output results/custom_features/training.json
```

This runs 20 all-active warmup epochs, fixes a reproducible random 50% subset of
nodes as shared operators, then runs 20 fine-tuning epochs. The JSON includes
dimensions, sample counts, seed, losses and held-out accuracies for both phases.
The random selector illustrates the API, not the paper's task-aware exact-K
selection. For multiple tasks, adapt the task loaders and task_classes in
train_mpu_pnn.py.

To create a tiny input only for software testing:

```bash
python examples/train_custom_features.py --input results/custom_features/toy.npz --make-demo
```

This generates 32 training and 16 test vectors with 8 features, using seed 44.
It refuses to overwrite an existing input. These are software-test examples,
not experimental data or a paper benchmark. Runtime on real data depends on
dimension, samples and epochs; the tiny example takes seconds on CPU.
