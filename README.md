# Bert-CRF

This is a PyTorch implementation of Bert-CRF model in PyTorch. 

## Introduction

Recent advances in transfer learning in NLP domain have facilitated the use of learned representations in a multitude of downstream NLP task. For sequence labeling tasks,
where availability of tagged data is major roadblock, use of pre-trained models have shown promising results. Traditionally, CRFs have been used for sequence labeling tasks
with manually handcrafted features. In this repository, Bert-CRF have been implemented to utilise the learned representation of pre-trained Bert model and structured use of 
previous tag information by CRF. Hugging Face implementation of Bert has been used in this repository, hence theoritically transfer learning model can be replaced with any suitable model.

## Dependencies

This implementation requires following Python 3.5 and following python libraries:

```
Transformers 3.5.1
PyTorch 1.3.0
tqdm 4.53.0
pickle 4.0
sklearn 0.19.1
```

## Procedure

### Data Preparation

The train, test and valid data should be formatted in a specified manner and kept in pickled form in ```./data/data_name``` folder. Each data file should be a 
list of input in following format:

```
[['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'],
  [2, 8, 1, 8, 8, 8, 1, 8, 8]]
 ```
 
 The tags should be label encoded using sklearn implementation and the LebelEncoder object should also be placed in ```./data/data_name``` folder.
 
 ### Model training
 
 The model training parameters can be configured in ```config.py``` file. Once the specified files are kept in location, execute the following command to initiate model training:
 
 ```sh
$ python main.py
```

## Issues/Pull Requests/Feedbacks

Don't hesitate to contact for any feedback or create issues/pull requests.
