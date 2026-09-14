import numpy as np

import h5py

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
    set the `role` attribute."""
    def __init_subclass__ (cls, name : str | None=None, role : str = "model",
                           fields=[], attrs=[], objects=[]):
        cls.stored_fields = set(fields)
        cls.stored_attrs = set(attrs)
        cls.stored_objects = set(objects)
        cls.model_name = name if name is not None else cls.__name__
        cls.role = role
        ModelFactory.register (cls.model_name, role, cls)
    @classmethod
    def h5load (cls, h5data):
        newself = cls()
        # slight inconsistency here: we just load all fields/attrs we find
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
                role = list(val.keys())[0]
                newself.__dict__[field] = load_model (val, role=role)
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
            self.__dict__[obj].h5save (g2)

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

class Factory:
    _registry = {}
    @classmethod
    def get (cls, name : str, h5data):
        try:
            return cls._registry[name].h5load(h5data)
        except KeyError:
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
    def __len__ (self):
        return self.M
    # TODO should we define vector_dimension and matrix_dimension here??



def h5info (h5file):
    from . import __version__
    grp = h5file.create_group("mctpy")
    grp.attrs['version'] = __version__

def load_model (h5file, role = "model"):
    h5grp = h5file[role]
    model_name = h5grp.attrs.get('model_name','')
    # we just try the factory, it will raise an exception if the model
    # cannot be loaded, user needs to catch that
    # empty string (if model_name is missing) will load loaded_model
    return ModelFactory.get(model_name, role, h5grp)

