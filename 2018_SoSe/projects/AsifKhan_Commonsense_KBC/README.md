## Commonsense Knowledge Base Reasoning

This repository implements course project as a part of NLP-Lab offered in Intelligent Systems track at University of Bonn.

This project implements neural network based models to score Knowledge Base tuples.
Here main focus is on commonsense knowledge base. In standard KB each triple is a tuple of form $(subject,predicate,object)$ and subject,predicate and object are represented by a unique token. In contrast to standard KB here subject and object are an arbitrary phrases thus represented by set of tokens.

For further detail on methods, evaluation setup and results refer to 
[report.ipynb](https://github.com/MdAsifKhan/NLP-Project/report.ipynb)

Trained Bilinear Averaging Model can be downloaded from [here](https://drive.google.com/file/d/1xbj9iD-Gw_8Y3j15ImwsEHkh_zT6JvAN/view?usp=sharing).
# Dataset
We use ConceptNet as a representation of commonsense knowledge. All data used in this experiment can be downloaded from: (http://ttic.uchicago.edu/~kgimpel/commonsense.html).


# Usage
The implementation is structured as follows.

1. ```model.py ```

Contains implementation of different neural networks for scoring a tuple. Currently we provide following models:
* Bilinear Averaging Model
* Bilinear LSTM Model
* DistMult Averaging Model
* DistMult LSTM Model
* ER-MLP Averaging Model
* ER-MLP LSTM Model

2. ```utils.py```

Implementation of preprocessing class, negative sampling and other basic utilities.
Main class and functions:
* class preprocess

Implements method to read arbitrary phrase ConceptNet triples and convert them to token representation for training neural network models. 

* function sample_negatives

Implements negative sampling strategy. Sampling is done by alternatively corrupting head and tail of a triple.

* class TripleDataset

Data class to support with pytorch batch loader.

3. ```evaluation.py```

Contains implementation of different evaluation metric. For this project we mainly use accuracy and auc score.


4. ```pretrained_embedding.py```

The scoring of tuple is highly dependent on initial embeddings used for training. To help model to better capture commonsense knowledge of ConceptNet we use pretraining. We create training data by combining ConceptNet tuples with natural language sentences of Open Mind Common Sense. 

5. ```run_experiment.py```

main file to evaluate neural network model for commonsense knowledge base completion. All data must be in a folder ```data/ConceptNet/``` . To run the experiment parameters need to be specified by white space tuples ex:
```
python run_experiment.py --model BilinearAvg --train_file train100k.txt \
		--valid_file dev1.txt --test_file dev2.txt --rel_file rel.txt \
		--pretrained_weights_file embeddings.txt --k 150 --dropout_p 0.2 \
		--mlp_hidden 100 --mb_size 200 --negative_samples 3 --nm_epoch 100 --lr 0.01 --lr_decay 1e-3 --weight_decay 1e-3 --embeddings_lambda 1e-2 
```

6. ```test_ConceptNet.py```

Evaluate model on test set. Trained model should be in a path as: 'models/ConceptNet/model.bin'
```
python test_ConceptNet.py --model BilinearAvg --test_file 'dev2.txt' \
		--rel_file 'rel.txt' --pretrained_weights_file 'embeddings.txt' --k 150 --dropout_p 0.2 --mb_size 200 --negative_samples 3 --nm_epoch 100 --lr 0.01 --lr_decay 1e-3 --weight_decay 1e-3 --embeddings_lambda 1e-2 
```

7. ```evaluate_topfive.py```

Realtime evaluation of arbitrary phrase. ex:

```
python evaluate_topfive.py sub drive_fast pred accident eval_type topfive
```



# Requirements
1. Pytorch
2. Python3+

# Comments & Feedback
For any comment or feedback please contact Mohammad Asif Khan via mail.