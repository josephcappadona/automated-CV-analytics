import itertools
from sklearn import model_selection

" Creates a `model_args` dict for every possible combination of parameters "
" Assumes that each parameter that is a list represents multiple possible values for that argument "
def get_arg_combos(model_args):

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

def get_folds(X, y, **kwargs):
    skf = model_selection.StratifiedKFold(**kwargs)
    for train_indices, test_indices in skf.split(X, y):
        yield (X[train_indices], y[train_indices]), (X[test_indices], y[test_indices])

