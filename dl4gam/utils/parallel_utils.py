import gc
import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

DEFAULT_NUM_PROCS = 1
DEFAULT_PBAR = False

log = logging.getLogger(__name__)


def set_default_num_procs(num_procs):
    """Set the default number of processes to use in parallel processing"""
    global DEFAULT_NUM_PROCS
    DEFAULT_NUM_PROCS = num_procs


def set_default_pbar(pbar):
    """Set the default progress bar to use in parallel processing"""
    global DEFAULT_PBAR
    DEFAULT_PBAR = pbar


def _fn_star(kwargs):
    fun = kwargs['fun']
    kwargs_others = {k: v for k, v in kwargs.items() if k != 'fun'}

    return fun(**kwargs_others)


def run_in_parallel(
        fun,
        num_procs=None,
        pbar=None,
        pbar_desc=None,
        gc_collect_step=None,
        continue_on_error=False,
        **kwargs
):
    """Wraps the function `fun` to run it in parallel using multiple processes.

    The function can take multiple arguments, some of which can be lists of the same length.
    The function will be called with each combination of the arguments, and the results will be collected in a list.
    For the other arguments that are not lists, the function will be called with the same value for each combination
    (tuples are also considered constant).

    :param fun: the function to run in parallel; it should accept keyword arguments
    :param num_procs: the number of processes to use for parallel run; if None, we use the default global value if set
    :param pbar: whether to show a progress bar; if None, the default global value will be used if set
    :param pbar_desc: the description for the progress bar; if None, it will be set to the function name and num_procs
    :param gc_collect_step: if provided, the garbage collector will be called every `gc_collect_step` iterations
    :param continue_on_error: if True, the function will continue running even if some calls fail; the results for
        the failed calls will be None; if False, the function will raise an exception on the first error encountered
    :param kwargs: the keyword arguments to pass to the function
    :return: a list of results from the function calls, in the same order as the input arguments
    """

    # use the global default values if not provided
    if num_procs is None:
        num_procs = DEFAULT_NUM_PROCS
    if pbar is None:
        pbar = DEFAULT_PBAR

    # check if the arguments which are lists have the same length
    arg_lens = [len(x) for _, x in kwargs.items() if isinstance(x, list)]
    if len(arg_lens) == 0:
        raise ValueError(
            'No arguments provided as lists. '
            'To run the function in parallel, at least one argument should be a list.'
        )

    # Check if all lists have the same length
    if max(arg_lens) != min(arg_lens):
        name_to_len = {k: len(v) for k, v in kwargs.items() if isinstance(v, list)}
        raise ValueError(f"Arguments provided as lists have different lengths: {name_to_len}")

    # repeat the arguments which are not lists
    kwargs['fun'] = fun
    kwargs_for_zip = {k: v if isinstance(v, list) else itertools.repeat(v) for k, v in kwargs.items()}
    kwargs_flatten = [{k: x for k, x in zip(kwargs_for_zip.keys(), y)} for y in zip(*kwargs_for_zip.values())]

    # run and collect the results
    n_tasks = len(kwargs_flatten)
    if pbar_desc is None:
        fun_name = fun.func.__name__ if hasattr(fun, 'func') else fun.__name__
        pbar_desc = f'Running {fun_name} with {num_procs} process(es)'

    if num_procs > 1:
        all_res = [None] * n_tasks
        with ProcessPoolExecutor(max_workers=num_procs) as executor:
            futures = {}  # features -> index, parameters
            iterator = list(enumerate(kwargs_flatten))  # task index, parameters

            # Submit initial batch
            tasks_submitted = 0
            for _ in range(min(num_procs, n_tasks)):
                i_task, kw = iterator[tasks_submitted]
                future = executor.submit(_fn_star, kw)
                futures[future] = (i_task, kw)
                tasks_submitted += 1

            with tqdm(total=len(kwargs_flatten), desc=pbar_desc, disable=not pbar) as pbar:
                while futures:
                    for future in as_completed(futures):
                        i_task, kw = futures.pop(future)

                        try:
                            all_res[i_task] = future.result()  # place the result in the correct index
                        except Exception as e:
                            log.error(f"Error occurred while processing the input: {kw}")
                            if continue_on_error:
                                all_res[i_task] = None
                            else:
                                raise e

                        pbar.update()

                        # Clean up memory once in a while
                        if gc_collect_step and pbar.n % gc_collect_step == 0:
                            gc.collect()

                        # Submit next task
                        if tasks_submitted < n_tasks:
                            idx, kw = iterator[tasks_submitted]
                            future = executor.submit(_fn_star, kw)
                            futures[future] = (idx, kw)
                            tasks_submitted += 1

                        break  # back to waiting to next completed future
    else:
        all_res = []
        with tqdm(total=arg_lens[0], desc=pbar_desc, disable=not pbar) as pbar:
            for crt_args in kwargs_flatten:
                try:
                    res = _fn_star(crt_args)
                except Exception as e:
                    log.error(f"Error occurred while processing the input: {crt_args}")
                    if continue_on_error:
                        res = None
                    else:
                        raise e
                all_res.append(res)
                pbar.update()

    return all_res
