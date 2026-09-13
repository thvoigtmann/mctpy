import numpy as np

from .__util__ import model_base

# for the registry mechanism, see
# https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
class LoadableModel:
    r"""Loadable model base class.

    This implements a basic save/load mechanism into h5 data files
    for models. In order to implement this for a given model, derive
    this model both from `model_base` and from `LoadableModel`,
    and set `fields` and `attrs` to lists containing the names of
    the object data fields that you need to store and restore.
    (The splitting into fields and attributes follows the usual guideline
    for h5 files: larger data arrays should be stored as fields,
    single values and small chunks can be stored as attributes.)
    Then also implement a `reinit` method that can be called without
    any parameters once upon loading all the fields and attributes have
    been set, to perform calculation of cached data that is not stored.

    Using this base class automatically registers the model in a global
    registry, so that the proper model type can be instantiated upon
    loading. The name will be the class name by default, but can be
    overridden by the `name` attribute."""
    def __init_subclass__ (cls, name : str | None=None, fields=[], attrs=[]):
        cls.stored_fields = set(fields)
        cls.stored_attrs = set(attrs)
        cls.model_name = name if name is not None else cls.__name__
        ModelFactory.register (cls.model_name, cls)
    @classmethod
    def h5load (cls, h5data):
        newself = cls()
        for attr,val in h5data.attrs.items():
            newself.__dict__[attr] = val
        for field,val in h5data.items():
            newself.__dict__[field] = np.array(val)
        newself.reinit()
        return newself
    def h5save (self, h5file):
        h5grp = h5file.create_group('model')
        h5grp.attrs['model_name'] = self.model_name
        for field in self.stored_fields:
            h5grp.create_dataset(field, data=self.__dict__[field])
        for attr in self.stored_attrs:
            if not self.__dict__[attr] is None:
                h5grp.attrs[attr] = self.__dict__[attr]

class ModelFactory:
    _registry = {}
    @classmethod
    def get (cls, name : str, h5data):
        try:
            return cls._registry[name].h5load(h5data)
        except KeyError:
            raise ValueError(f"unknown model : {name}")
    @classmethod
    def register (cls, name : str, loadable_cls : LoadableModel):
        cls._registry[name] = loadable_cls

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

def load_model (h5file):
    h5grp = h5file['model']
    model_name = h5grp.attrs.get('model_name','')
    # we just try the factory, it will raise an exception if the model
    # cannot be loaded, user needs to catch that
    # empty string (if model_name is missing) will load loaded_model
    return ModelFactory.get(model_name, h5grp)

