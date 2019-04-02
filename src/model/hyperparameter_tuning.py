import logging
import itertools
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
        model_args_ = model_args.copy()
        for arg_name, arg_val in arg_combo:
            if type(arg_val) != tuple:
                model_args_[arg_name] = arg_val
            else:
                subarg_name, subarg_val = arg_val
                model_args_[arg_name][subarg_name] = subarg_val
        model_args_list.append(model_args_)

    return model_args_list

def find_best_model(model_configs, build_model, X, y):

    if len(model_configs) > 1:
        # find best model config with k-fold cross validation
        logging.info('%d possible model configs.' % len(model_configs))
        logging.info('Finding best model config with k-fold cross validation...')

        avg_errors = []
        for i, model_config in enumerate(model_configs):

            print('\n'); logging.info('Model #%d of %d:' % (i+1, len(model_configs)))
            logging.info('\n' + pprint.pformat(model_config))

            n_folds = model_config['n_folds']
            cross_val_errors = []
            for i, ((X_train, y_train), (X_test, y_test)) \
                in enumerate(get_folds(X, y, n_splits=n_folds)):

                print(); logging.info('Fold #%d of %d' % (i+1, n_folds))

                _, val_error = build_model(model_config, X_train, y_train, X_test, y_test)
                cross_val_errors.append(val_error)

            avg_error = sum(cross_val_errors) / len(cross_val_errors)
            avg_errors.append(avg_error)
            print(); logging.info('Average cross validation error: %g' % avg_error)

        best_error = min(avg_errors)
        error_infos = list(zip(model_configs, avg_errors))
        best = list(filter(lambda m_e: m_e[1] == best_error, error_infos))
        print('\n'); logging.info('%d model config(s) had the best average cross validation error (%g)' % (len(best), best_error))

        if len(best) > 1:
            logging.info('Picking best model config randomly...')
            best_config, _ = random.choice(best)
        else:
            best_config, _ = best[0]
        logging.info('Chosen config: \n%s' % pprint.pformat(best_config))

        print(); logging.info('Building model with best config on all training data...')
        model, _ = build_model(best_config, X, y, X, y)

    else:
        # Only one possible model config
        config = model_configs[0]
        model, validation_error = build_model(config, X, y, X, y)
        error_infos = [(config, validation_error)]
    return model, error_infos

def get_folds(X, y, **kwargs):
    skf = model_selection.StratifiedKFold(**kwargs)
    for train_indices, test_indices in skf.split(X, y):
        yield (X[train_indices], y[train_indices]), (X[test_indices], y[test_indices])

