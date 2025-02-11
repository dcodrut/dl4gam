import multiprocessing
import argparse
from tqdm import tqdm
import itertools


def str2bool(v):
    # https://stackoverflow.com/a/43357954
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _fn_star(kwargs):
    fun = kwargs['fun']
    kwargs_others = {k: v for k, v in kwargs.items() if k != 'fun'}
    return fun(**kwargs_others)


def run_in_parallel(fun, num_procs=1, pbar=False, pbar_desc=None, **kwargs):
    # check if the arguments which are lists have the same length
    arg_lens = [len(x) for _, x in kwargs.items() if isinstance(x, list)]
    assert len(arg_lens) > 0, 'At least one argument is expected to be a list.'
    assert max(arg_lens) == min(arg_lens), 'The arguments provided as lists should have the same length'

    # repeat the arguments which are not lists
    kwargs['fun'] = fun
    kwargs_for_zip = {k: v if isinstance(v, list) else itertools.repeat(v) for k, v in kwargs.items()}
    kwargs_flatten = [{k: x for k, x in zip(kwargs_for_zip.keys(), y)} for y in zip(*kwargs_for_zip.values())]

    # run and collect the results
    all_res = []
    if pbar_desc is None:
        fun_name = fun.func.__name__ if hasattr(fun, 'func') else fun.__name__
        pbar_desc = f'Running {fun_name} with {num_procs} process(es)'
    with tqdm(total=arg_lens[0], desc=pbar_desc, disable=not pbar) as pbar:
        if num_procs > 1:
            with multiprocessing.Pool(num_procs) as pool:
                for res in pool.imap_unordered(_fn_star, kwargs_flatten):
                    all_res.append(res)
                    pbar.update()
        else:
            for crt_args in kwargs_flatten:
                res = _fn_star(crt_args)
                all_res.append(res)
                pbar.update()
    return all_res
