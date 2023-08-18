# Distribuirani ML

  Ova praksa baviti će se izradom primjera koji će se baviti distribuiranim ML
  proračunom koristeći poznate knjižnice poput scikit-learn i TensorFlow.
  Algoritmi koji se tipično koriste u svrhu klasifikacije putem strojnog učenja
  prilagoditi će se okolini superračunala Supek na kojem će se iskoristiti
  mogućnost raspodjele računa na više procesora i čvorova.

  Algoritmi i knjižnice koje će se pritom koristiti su:
  - [xgboost](https://xgboost.readthedocs.io/en/stable/)
  - [ResNet50](https://towardsdatascience.com/the-annotated-resnet-50-a6c536034758) korištenjem [TensorFlowa](https://www.tensorflow.org/api_docs)

## Zadaci

  1. Implementirati [xgboost model](https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/) za klasifikaciju opažanja [Higgsovog bozona](https://archive.ics.uci.edu/dataset/280/higgs). Pri razvoju, oslanjati se na knjižnicu [dask](https://xgboost.readthedocs.io/en/stable/tutorials/dask.html) za dostavljanje podataka, traženje hiperparametara i treniranje modela.
  2. Implementirati [Resnet50 model](https://www.kaggle.com/code/suniliitb96/tutorial-keras-transfer-learning-with-resnet50) za predviđanje starosti lica na temelju dataseta [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/?ref=hackernoon.com). Pri razvoju, oslanjati se na knjižnicu [Ray](https://docs.ray.io/en/releases-1.11.0/ray-core/using-ray-with-tensorflow.html) za dostavljanje podataka, traženje hiperparametara i treniranje modela. 

## Linkovi

  - Supek
      - [Javni wiki](https://wiki.srce.hr/display/NR)
      - [Korištenje python knjižnica na Supeku](https://wiki.srce.hr/display/NR/Python%2C+pip+i+conda)
      - [Dask](https://wiki.srce.hr/display/NR/Dask)
      - [TensorFlow](https://wiki.srce.hr/display/NR/TensorFlow)
  - xgboost & Higgs & dask
      - [Tutorial 1](https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/)
      - [Tutorial 2](https://www.datacamp.com/tutorial/xgboost-in-python)
      - [GitHub jednog od rješenja](https://github.com/andyh47/higgs)
      - [Higgs članak](https://proceedings.mlr.press/v42/chen14.pdf)
  - TensorFlow & Ray
      - [ResNet50 Tutorial 1](https://www.kaggle.com/code/suniliitb96/tutorial-keras-transfer-learning-with-resnet50)
      - [ResNet50 Tutorial 2](https://github.com/ovh/ai-training-examples/blob/main/notebooks/computer-vision/image-classification/tensorflow/resnet50/notebook-resnet-transfer-learning-image-classification.ipynb)
      - [TensorFlow + Ray Train ](https://docs.ray.io/en/latest/train/examples/tf/tensorflow_mnist_example.html#tensorflow-mnist-example)
      - [TensorFlow + Ray Tune](https://docs.ray.io/en/latest/tune/examples/tune_mnist_keras.html#tune-mnist-keras)
