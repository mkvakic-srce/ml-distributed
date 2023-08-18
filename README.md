# Distribuirani ML

  Ova praksa baviti će se izradom primjera koji će se baviti distribuiranim ML
  proračunom koristeći poznate knjižnice poput scikit-learn, TensorFlow i
  PyTorch. Algoritmi koji se tipično koriste u svrhu klasifikacije putem
  strojnog učenja prilagoditi će se okolini superračunala Supek na kojem će se
  iskoristiti mogućnost raspodjele računa na više procesora i čvorova.

  Algoritmi i knjižnice koje će se pritom koristiti su:
  - [xgboost](https://xgboost.readthedocs.io/en/stable/)
  - [ResNet50]() korištenjem TensorFlowa
  - [BERT] korištenjem HuggingFacea

## Zadaci

  1. Implementirati [xgboost model](https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/) za klasifikaciju opažanja [Higgsovog bozona](https://archive.ics.uci.edu/dataset/280/higgs). Pri razvoju, oslanjati se na knjižnicu [dask](https://xgboost.readthedocs.io/en/stable/tutorials/dask.html) za dostavljanje podataka, traženje hiperparametara i treniranje modela.
  2. Implementirati [Resnet50 model](https://www.kaggle.com/code/suniliitb96/tutorial-keras-transfer-learning-with-resnet50) za predviđanje starosti lica na temelju dataseta [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/?ref=hackernoon.com). Pri razvoju, oslanjati se na knjižnicu [Ray](https://docs.ray.io/en/releases-1.11.0/ray-core/using-ray-with-tensorflow.html) za dostavljanje podataka, traženje hiperparametara i treniranje modela. 
  3. Prilagoditi postojeći PyTorch kod za [analizu sentimenta](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/6%20-%20Transformers%20for%20Sentiment%20Analysis.ipynb) za izvođenje na Supeku

## Linkovi

  - Supek
      - [Javni wiki](https://wiki.srce.hr/display/NR)
      - [Python knjižnice na Supeku](https://wiki.srce.hr/display/NR/Python%2C+pip+i+conda)
      - [Dask](https://wiki.srce.hr/display/NR/Dask)
      - [TensorFlow](https://wiki.srce.hr/display/NR/TensorFlow)
      - [PyTorch](https://wiki.srce.hr/display/NR/PyTorch?src=contextnavpagetreemode)
  - xgboost & Higgs & dask
      - [Tutorial 1](https://www.datacamp.com/tutorial/xgboost-in-python)
      - [Tutorial 2]()
      - [Higgs članak](https://proceedings.mlr.press/v42/chen14.pdf)
      - [GitHub jednog od rješenja](https://github.com/andyh47/higgs)
  - TensorFlow & Resnet50 & Ray
      - [Tutorial 1](https://www.kaggle.com/code/suniliitb96/tutorial-keras-transfer-learning-with-resnet50)
      - [Tutorial 2](https://github.com/ovh/ai-training-examples/blob/main/notebooks/computer-vision/image-classification/tensorflow/resnet50/notebook-resnet-transfer-learning-image-classification.ipynb)
  - BERT & Huggingface & PyTorch
