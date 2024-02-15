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
| Original VAE | [original_vae.py](/models/original_vae.py) | MNIST |

Installation:
=================
* Python virual env
```bash
python3 -m pip install virtualenv

python3 -m venv env

source env/bin/activate

python3 -m pip install -r requirements.txt
```

* Docker
> Note: Update later

Training:
=================

* Example
```python
python3 train.py --model origin --dataset mnist --data_path ./ --batch_size 100 --epochs 30
```
> Note: Training results will be saved to `./results`