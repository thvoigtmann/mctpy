import h5py
import numpy as np

from .__h5__ import h5info, Factory, load_model

from .fsolver import nonergodicity_parameter, eigenvalue
from .solver import correlator

def load_file (filename):
    """Load MCT results from a h5 data format file.

    Parameters
    ----------
    filename : str
        File name to be read from.

    Result
    ------
    result : dict or mct_object
        If the loaded file contained a single mctpy object, return that;
        else return a nested dictionary of mctpy objects as defined by
        self.reinit()
        the user saving this file.
    """
    result = {}
    with h5py.File (filename, 'r') as f:
        _load_h5 (f, result, key='')
    if not (result.keys() - set([''])):
        # if we have no subkeys
        result = result['']
    return result

def _load_h5 (h5data, res, key):
    for grp in h5data.keys():
        if grp == 'mctpy':
            pass # could do checks here
        elif Factory.contains(grp):
            res[key] = Factory.get(grp, h5data)
        elif grp == 'model' or grp == 'structurefactor':
            res[grp] = load_model(h5data, role=grp)
        elif isinstance(h5data[grp], h5py.Dataset):
            # maybe emit warning: we are not really prepared
            # for free-style data floating around in the h5 yet
            res[grp] = np.array(h5data[grp])
        else:
            res[grp] = {}
            print("LOAD H5 grp",grp)
            _load_h5 (h5data[grp], res[grp], '')
            print("CHECK",res[grp].keys())
            if not (res[grp].keys() - set([''])):
                # if we have no subkeys
                res[grp] = res[grp]['']

def save_file (filename, items):
    """Save MCT results to a file in h5 data format.

    Parameters
    ----------
    filename : str
        File name to be written to, will be overwritten.
    items : dict or mct_object
        The data to be saved. This can either be a single mctpy object,
        or a nested dictionary of keys pointing to mctpy objects.
        Here, a save-able mctpy object is any object that implements
        a `h5save` method. This is intended for `mctpy.correlator`
        or `mctpy.nonergodicity_parameter` objects mostly.
    """
    with h5py.File (filename, 'w') as f:
        h5info(f)
        _save_h5 (f, items)

def _save_h5 (h5file, items):
    if isinstance(items,dict):
        for key, val in items.items():
            _save_h5 (h5file.create_group(str(key)), val)
    else:
        items.h5save (h5file)
