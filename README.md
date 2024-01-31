## VAE Models Implementations

# Some models supported:

| Model | Code | Dataset |
| --- | --- | --- |
| Original VAE | [OriginalVAE.py](/models/OriginalVAE.py) | MNIST |

# Installation:

```bash
python3 -m pip install -r requirements.txt
```

# Training:

```python
python3 train.py --model origin --dataset mnist --data_path ./ --batch_size 100 --epochs 30
```