import os
import sys

if sys.version_info[0] == 2:  # pragma: no cover
    from ConfigParser import RawConfigParser
else:
    from configparser import RawConfigParser

try:
    import dnnlib
except ImportError:
    dnnlib = None



_CONFIG_PATH = './semantic_segmentation.cfg'

_config__ = None

def get_config():  # pragma: no cover
    global _config__
    if _config__ is None:
        if os.path.exists(_CONFIG_PATH):
            try:
                _config__ = RawConfigParser()
                _config__.read(_CONFIG_PATH)
            except Exception as e:
                print('WARNING: error {} trying to open config '
                      'file from {}'.format(e, _CONFIG_PATH))
                _config__ = RawConfigParser()
        else:
            _config__ = RawConfigParser()
    return _config__


def get_config_dir(name, exists=True):  # pragma: no cover
    dir_path = get_config().get('paths', name)
    if exists:
        if not os.path.exists(dir_path):
            raise RuntimeError(
                'semantic_segmentation.settings: the directory path {} does not exist'.format(dir_path))
    return dir_path


def get_data_path(config_name, dnnlib_template, exists=True):
    if dnnlib is not None:
        return dnnlib.submission.submit.get_path_from_template(dnnlib_template)
    else:
        return get_config_dir(config_name, exists=exists)