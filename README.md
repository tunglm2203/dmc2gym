# OpenAI Gym wrapper for the DeepMind Control Suite.
A lightweight wrapper around the DeepMind Control Suite that provides the standard OpenAI Gym interface. The wrapper allows to specify the following:
* Reliable random seed initialization that will ensure deterministic behaviour.
* Setting ```from_pixels=True``` converts proprioceptive observations into image-based. In additional, you can choose the image dimensions, by setting ```height``` and ```width```.
* Action space normalization bound each action's coordinate into the ```[-1, 1]``` range.
* Setting ```frame_skip``` argument lets to perform action repeat.
* Download the [DAVIS 2017
  dataset](https://davischallenge.org/davis2017/code.html) and unzip to folder `./dmc2gym/videos/`.


### Instalation
Go to dmc2gym directory then type the command:
```
pip install -e .
```
or
```
python setup.py install
```

### Usage

See `test_env.py` file.
