# VAE Models Implementations

Table of contents
=================
<!--ts-->
   * [Models supported](#some-models-supported)
   * [Installation](#installation)
   * [Training](#training)
<!--te-->

Some models supported:
================

| Model | Code | Dataset |
| --- | --- | --- |
| Original VAE | [originalVAE.py](/models/originalVAE.py) | MNIST |

Installation:
=================
```bash
python3 -m pip install virtualenv

python3 -m venv env

source env/bin/activate

python3 -m pip install -r requirements.txt
```

Training:
=================
```python
python3 train.py --model origin --dataset mnist --data_path ./ --batch_size 100 --epochs 30
```
> Note: Results will be saved to `./results`