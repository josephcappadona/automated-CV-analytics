import logging
import time
import itertools
import copy
import pprint
import random
import utils
from sklearn import model_selection

" Creates a `model_args` dict for every possible combination of parameters "
" Assumes that each parameter that is a list represents multiple possible values for that argument "
def get_model_configs(model_args):

    l = []
    for arg_name, arg_val in model_args.items():

        if type(arg_val) == list:
            l.append(list(itertools.product([arg_name], arg_val)))
        elif type(arg_val) == dict:
            for subarg_name, subarg_val in arg_val.items():
                if type(subarg_val) == list:
                    l.append(list(itertools.product([arg_name], list(itertools.product([subarg_name], subarg_val)))))

    if not l: # only one possible parameter configuration
        return [model_args]
    args_combos = list(itertools.product(*l))

    model_args_list = []
    for arg_combo in args_combos:
        model_args_ = copy.deepcopy(model_args)
        for arg_name, arg_val in arg_combo:
            if type(arg_val) != tuple:
                model_args_[arg_name] = arg_val
            else:
                subarg_name, subarg_val = arg_val
                model_args_[arg_name][subarg_name] = subarg_val
        model_args_list.append(model_args_)
    return model_args_list


def find_best_models(all_models):
    best_test_acc = max(all_models, key=lambda x: x[3])[3]  # get elmt with highest test_acc
    best_models = [(m,c,cv_e,t_a) for m,c,cv_e,t_a in all_models if t_a == best_test_acc]
    return best_models

def create_all_models(model_configs, build_model, cross_val_model, test_model, X_train, y_train, X_test, y_test):

    res = []
    n_configs = len(model_configs)
    for i, model_config in enumerate(model_configs):
        print_model_details(model_config, i+1, n_configs)
        
        # limit the number of training examples
        X_train_, y_train_ = X_train[:], y_train[:]
        if 'n_train_examples' in model_config:
            n_train_examples = model_config['n_train_examples']
            logging.info('Limiting # of positive training examples to %d per class...' % n_train_examples)
            X_train_, y_train_ = utils.remove_extra_examples(X_train_, y_train_, n_train_examples)

        #
        if 'n_folds' in model_config:
            logging.info('Computing average cross-validation error...')
            val_error = cross_val_model(model_config, X_train_, y_train_)

        #
        logging.info('Building full model...')
        full_model = build_model(model_config, X_train_, y_train_)

        #
        if 'n_folds' not in model_config:
            _, val_error = test_model(full_model, X_train_, y_train_, test_type='Validation')

        #
        print(); logging.info('Testing full model...')
        test_acc, _ = test_model(full_model, X_test, y_test)

        res.append((full_model, model_config, val_error, test_acc))

    return res

def print_model_details(model_config, i, n_configs):
    print('\n\n')
    logging.info('Model #%d of %d:' % (i, n_configs))
    logging.info('\n' + pprint.pformat(model_config))

def get_folds(X, y, **kwargs):
    skf = model_selection.StratifiedKFold(**kwargs)
    for train_indices, test_indices in skf.split(X, y):
        yield (X[train_indices], y[train_indices]), (X[test_indices], y[test_indices])

