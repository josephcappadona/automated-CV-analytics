from collections import defaultdict

def parse_args(args):
    args_dict = defaultdict(str)
    for i, arg in enumerate(args):
        if arg[0] == '-':
            prefix = arg.strip('-')
            suffix = args[i+1]
            args_dict[prefix] = suffix
    return args_dict
