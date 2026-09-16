import numpy as np

import h5py
import sys

from .__util__ import model_base

# for the registry mechanism, see
# https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
class LoadableModel:
    r"""Loadable model base class.

    This implements a basic save/load mechanism into h5 data files
    for models. In order to implement this for a given model, derive
    this model both from `model_base` and from `Loadable`,
    and set `fields` and `attrs` to lists containing the names of
    the data fields that you need to store and restore.
    (The splitting into fields and attributes follows the usual guideline
    for h5 files: larger data arrays should be stored as fields,
    single values and small chunks can be stored as attributes.)
    It is also possible to store further nested `objects` that will
    recursively call their own save/load mechanisms.
    Then also implement a `reinit` method that can be called without
    any parameters once upon loading all the fields and attributes have
    been set, to perform calculation of cached data that is not stored.

    Using this base class automatically registers the model in a global
    registry, so that the proper model type can be instantiated upon
    loading. The name will be the class name by default, but can be
    overridden by the `name` attribute.

    The same mechanism is also used for structure factors; for this,
    set the `role` attribute. But see the separate class `Loadable`
    for the registry used for main container objects such as
    `mctpy.correlator` or `mct.eigenvalue`."""
    def __init_subclass__ (cls, name : str | None=None, role : str = "model",
                           fields=[], attrs=[], objects=[],
                           resetable=[], safe_reset=[]):
        """Register the model in the factory used by the HDF5 loader,
        under the given `name` in the given `role`.
        The `fields`, `attrs`, and `objects` can be used to designate
        the fields that will be written to and read from files,
        in the form of HDF5 datasets, attributes, and custom groups,
        respectively.
        The `resetable` set indicates those variables that can be
        reset using the `set` method, allowing to quickly change one
        out of a few model parameters. This will usually require a
        recalculation of vertices, unless the parameter is listed in
        the `safe_reset` list.
        """
        cls.stored_fields = set(fields)
        cls.stored_attrs = set(attrs)
        cls.stored_objects = set(objects)
        cls.resetable_fields = set(resetable)
        cls.safe_reset = set(safe_reset)
        cls.model_name = name if name is not None else cls.__name__
        cls.role = role
        ModelFactory.register (cls.model_name, role, cls)
    def set (self, /, safe=[], **kwargs):
        """Method to reset model parameters.

        Parameters
        ----------
        kwargs : kwargs
            The allowed parameters to reset are given in the classes'
            `resetable_fields`. All those that are not also listed as
            `safe_reset` for the class will trigger a call to `reinit`,
            i.e. potentially a vertex recalculation.
        """
        need_reinit = False
        for k,v in kwargs.items():
            if k in self.resetable_fields:
                self.__dict__[k] = v
            if not k in self.safe_reset:
                need_reinit |= True
        if need_reinit:
            self.reinit()
    @classmethod
    def h5load (cls, h5data):
        # create a new uninitialized object
        # will will fill the data fields ourselves
        newself = cls.__new__(cls)
        super(cls,newself).__init__()
        # slight inconsistency here: we just load all fields/attrs we find
        # not the ones that the class might define to be saved
        # if we find a field that is not a dataset, we try to load it as
        # an object from the factory
        for attr,val in h5data.attrs.items():
            newself.__dict__[attr] = val
        for field,val in h5data.items():
            if isinstance(val, h5py.Dataset):
                newself.__dict__[field] = np.array(val)
            else:
                # for an object, we expect a hdf5 path of the form
                # .../field/role (e.g. .../Sq/structurefactor)
                # the load_model call will itself descend there
                roles = list(val.keys())
                if len(roles)==1:
                    newself.__dict__[field] = load_model (val, role=roles[0])
                # could issue a warning if none (happens if eg Sq wasnt savable)
                # or if too many (unexpected h5 file structure)
        newself.reinit()
        return newself
    def h5save (self, h5file):
        h5grp = h5file.create_group(self.role)
        h5grp.attrs['model_name'] = self.model_name
        for field in self.stored_fields:
            h5grp.create_dataset(field, data=self.__dict__[field])
        for attr in self.stored_attrs:
            if not self.__dict__[attr] is None:
                h5grp.attrs[attr] = self.__dict__[attr]
        for obj in self.stored_objects:
            g2 = h5grp.create_group(obj)
            if 'h5save' in dir(self.__dict__[obj]):
                self.__dict__[obj].h5save (g2)
            else:
                print(f"WARNING cannot save {obj} {self.__dict__[obj]}",
                      file=sys.stderr)

class ModelFactory:
    _registry = { 'model': {},
                  'structurefactor': {} }
    @classmethod
    def get (cls, name : str, role : str, h5data):
        try:
            return cls._registry[role][name].h5load(h5data)
        except KeyError:
            raise ValueError(f"unknown {role} : {name}")
    @classmethod
    def register (cls, name : str, role : str, loadable_cls : LoadableModel):
        cls._registry[role][name] = loadable_cls

class Loadable:
    """Loadable based object, used to register the core objects
    we use in our h5 data files, such as correlators, nonergodicity
    parameters and eigenvalues.

    All objects derived from this class are expected to implement
    a classmethod `h5load`."""
    def __init_subclass__ (cls, name : str | None=None, fields=[], attrs=[]):
        cls.stored_fields = set(fields)
        cls.stored_attrs = set(attrs)
        cls.model_name = name if name is not None else cls.__name__
        Factory.register (cls.model_name, cls)
    @classmethod
    def load (cls, file):
        """Load the object from a h5 file.

        Parameters
        ----------
        file : str
            File name to read the object from.
        """
        with h5py.File(file, 'r') as f:
            obj = cls.h5load(f)
        return obj
    def save (self, file):
        """Save the object to a h5 file.

        Parameters
        ----------
        file : str
            File name to write the object to, will be overwritten.
        """
        with h5py.File(file, 'w') as f:
            self.h5save(f)


class Factory:
    _registry = {}
    @classmethod
    def get (cls, name : str, h5data):
        try:
            return cls._registry[name].h5load(h5data)
        except KeyError as e:
            print(e)
            raise ValueError(f"unknown object: {name}")
    @classmethod
    def register (cls, name : str, loadable_cls : Loadable):
        cls._registry[name] = loadable_cls
    @classmethod
    def contains (cls, name : str):
        return name in cls._registry


class loaded_model(model_base, LoadableModel, name=""):
    """Empty fallback class for models that have been loaded but are
    not recognized (for example because they did not register as a
    `LoadableModel` class."""
    def __init__ (self):
        self.M = 0
    def reinit (self):
        pass
    def __len__ (self):
        return self.M
    # TODO should we define vector_dimension and matrix_dimension here??



def h5info (h5file):
    from . import __version__
    grp = h5file.create_group("mctpy")
    grp.attrs['version'] = __version__

def load_model (h5file, role = "model"):
    if not role in h5file.keys():
        return None
    h5grp = h5file[role]
    model_name = h5grp.attrs.get('model_name','')
    # we just try the factory, it will raise an exception if the model
    # cannot be loaded, user needs to catch that
    # empty string (if model_name is missing) will load loaded_model
    return ModelFactory.get(model_name, role, h5grp)

