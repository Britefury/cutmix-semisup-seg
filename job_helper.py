import os
import inspect

import sys, re

LOG_PREFIX = re.compile('log_(\d+)')
JOB_DIR_PREFIX = re.compile('(\d+)')


class LogAlreadyExistsError (Exception):
    pass


class Logger (object):
    def __init__(self, path, stream):
        self.path = path
        self.stream = stream

    def write(self, x):
        with open(self.path, 'a+') as f_out:
            f_out.write(x)
        self.stream.write(x)

    def flush(self):
        self.stream.flush()


class SubmitConfig (object):
    def __init__(self, job_name, job_desc, enumerate_job_names):
        res_dir = os.path.join('results', job_name)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir, exist_ok=True)

        if job_desc == 'none':
            log_path = None
            job_out_dir = None
        else:
            if enumerate_job_names:
                job_index = 0
                res_dir_contents = os.listdir(res_dir)
                for name in res_dir_contents:
                    m = LOG_PREFIX.match(name)
                    if m is not None:
                        job_index = max(job_index, int(m.group(1)) + 1)
                    m = JOB_DIR_PREFIX.match(name)
                    if m is not None:
                        job_index = max(job_index, int(m.group(1)) + 1)

                log_path = os.path.join(res_dir, 'log_{:04d}_{}.txt'.format(job_index, job_desc))
                job_out_dir = os.path.join(res_dir, '{:04d}_{}'.format(job_index, job_desc))
            else:
                log_path = os.path.join(res_dir, 'log_{}.txt'.format(job_desc))
                job_out_dir = os.path.join(res_dir, job_desc)

                if os.path.exists(log_path) or os.path.exists(job_out_dir):
                    raise LogAlreadyExistsError

        self.log_path = log_path
        self.job_out_dir = job_out_dir
        # Run-dir created on the fly
        self.__run_dir = None

        if self.log_path is not None:
            self.__stdout = Logger(self.log_path, sys.stdout)
            self.__stderr = Logger(self.log_path, sys.stderr)

    @property
    def run_dir(self):
        if self.__run_dir is None and self.job_out_dir is not None:
            # Make the run dir to receive output
            self.__run_dir = self.job_out_dir
            os.makedirs(self.__run_dir, exist_ok=True)
        return self.__run_dir

    def connect_streams(self):
        if self.log_path is not None:
            sys.stdout = self.__stdout
            sys.stderr = self.__stderr

    def disconnect_streams(self):
        if self.log_path is not None:
            sys.stdout = self.__stdout.stream
            sys.stderr = self.__stderr.stream


def job(job_name, enumerate_job_names=True):
    """
    Decorator to turn a function into a job submitter.

    Usage:

    >>> @job('wait_some_time')
    ... def wait_some_time(submit_config: SubmitConfig, iteration_count):
    ...     fn = os.path.join(submit_config.run_dir, "output.txt")
    ...     with open(fn, 'w') as f:
    ...         f.write("Works!")
    ...
    ...     print('Training...')
    ...     for i in range(iteration_count):
    ...         if ctx.should_stop():
    ...             break
    ...
    ...         time.sleep(1.0)

    To submit a job:
    >>> wait_some_time.submit(on='local', job_desc='description_to_identify_specific_job', iteration_count=50)

    :param job_name: The name to be given to the job
    :param enumerate_job_names: Enumerated job name prefix
    """

    def decorate(job_fn):
        def run_job(**kwargs):
            specific_job_name = kwargs.pop('job_name', None)
            if specific_job_name == '':
                specific_job_name = None

            if specific_job_name is None:
                specific_job_name = job_name

            quota_group = kwargs.pop('quota_group', None)
            if quota_group is not None and quota_group != '':
                raise ValueError('quota_group not supported when dnnlib is not available')

            job_desc_arg = kwargs.pop('job_desc', None)
            if job_desc_arg is None or job_desc_arg == '':
                job_desc_arg = specific_job_name

            try:
                submit_config = SubmitConfig(specific_job_name, job_desc_arg, enumerate_job_names)
            except LogAlreadyExistsError:
                print('Job {}:{} already executed; skipping'.format(specific_job_name, job_desc_arg))
            else:
                print('[NO dnnlib] logging to {}'.format(submit_config.log_path))

                submit_config.connect_streams()
                try:
                    job_fn(submit_config, **kwargs)
                finally:
                    submit_config.disconnect_streams()


        job_fn.submit = run_job
        return job_fn

    return decorate
