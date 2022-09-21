import pandas as pd
import sklearn
import logging
import torch
import datetime
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from functools import partial
import sklearn
from scipy.special import softmax
import numpy as np

# set the logger to log the error level logs 
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)
# checking for cuda availability 
cuda_available = torch.cuda.is_available()
print("Is cuda available?", cuda_available)


project = 'name of project'
is_temporal = True
path_read = 'path_to_the folder which contains projects'
path_save = 'path to save the data'
output_name = 'path to save the output'
cuda_device = 0
'''learning model args'''
models = [ {'name': 'roberta', 'ver': 'roberta-base'}] #add model name
lr = 3e-5 
drp = 0
epochs = 6
batch_t = 8
batch_e = 8
max_seq = 512

# function to load the data
def load_data(project_name):
    if is_temporal == False:
        train_df = pd.read_csv(f'{path_read}/{project_name}/train_rand.csv')
        valid_df = pd.read_csv(f'{path_read}/{project_name}/valid_rand.csv')
        test_df = pd.read_csv(f'{path_read}/{project_name}/test_rand.csv')
    else:
        train_df = pd.read_csv(f'{path_read}/{project_name}/train_temp.csv')
        valid_df = pd.read_csv(f'{path_read}/{project_name}/valid_temp.csv')
        test_df = pd.read_csv(f'{path_read}/{project_name}/test_temp.csv')
    return train_df, valid_df, test_df

def calc(p1, p2, func, **kwargs):
    return func(p1, p2, **kwargs)

# calculating scores of model output
def get_max_scores(labels, model_outputs):
    probabilities = softmax(model_outputs, axis=1)
    prob_class_positive = [item[0] for item in probabilities]
    scores_dict ={'max_p': 0, 'max_r': 0, 'max_f1': 0, 'max_acc': 0, 'perf_t': 0}
    for tr in np.arange(0.05, 0.99, 0.05):
        predicted_label = [0 if item > tr else 1 for item in prob_class_positive]
        p = sklearn.metrics.precision_score(labels, predicted_label, average='binary')
        r = sklearn.metrics.recall_score(labels, predicted_label, average='binary')
        f = sklearn.metrics.f1_score(labels, predicted_label, average='binary')
        acc = sklearn.metrics.accuracy_score(labels, predicted_label)
        if f >= scores_dict['max_f1']:
            scores_dict['max_p'] = round(p, 3)
            scores_dict['max_r'] = round(r, 3)
            scores_dict['max_f1'] = round(f, 3)
            scores_dict['max_acc'] = round(acc, 3)
            scores_dict['pref_t'] = round(tr, 2)   
    return scores_dict

# metrics used for evaluation
metrics_recom = {
    "acc_norm": partial(calc,func=sklearn.metrics.accuracy_score) ,
    "p_bin": partial(calc,func=sklearn.metrics.precision_score,average='binary'),
    "r_bin": partial(calc,func=sklearn.metrics.recall_score,average='binary'),
    "f_bin": partial(calc,func=sklearn.metrics.f1_score,average='binary')
}

# creating the model based on the model args
def create_model(name, ver, lr, drp, epochs, batch_t, batch_e, max_seq):
    model_args = ClassificationArgs()
    model_name = name
    model_version = ver
    model_args.learning_rate = lr
    model_args.num_train_epochs = epochs
    model_args.eval_batch_size = batch_t
    model_args.train_batch_size = batch_e
    model_args.max_seq_length = max_seq
    model_args.config = {"classifier_dropout_prob": drp}
    model_args.output_dir = output_name +'/'
    model_args.overwrite_output_dir = True
    model_args.use_early_stopping = True
    model_args.early_stopping_delta = 0.01
    model_args.early_stopping_metric = "mcc"
    model_args.early_stopping_metric_minimize = False
    model_args.early_stopping_patience = 5
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 1000
    model_args.save_steps = -1
    model_args.save_model_every_epoch = False
    model_args.no_cache = True
    model = ClassificationModel(model_name, model_version, args = model_args, cuda_device = cuda_device, use_cuda=cuda_available)
    
    return model

# loading the data to three seperated data frame of train, valid, test
train_df, valid_df, test_df = load_data(project)

for model in models:
    name = model['name']
    ver = model['ver']
    model = create_model(name, ver, lr, drp, epochs, batch_t, batch_e, max_seq)   
    (t1, t2) = model.train_model(train_df=train_df, eval_df=valid_df, acc=sklearn.metrics.accuracy_score)
    results, model_outputs, wrong_pred = model.eval_model(test_df, verbose=True, **metrics_recom)
    max_scores = get_max_scores(test_df['labels'], model_outputs)        
