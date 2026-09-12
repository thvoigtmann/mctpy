import numpy as np

from .__util__ import model_base

# for the registry mechanism, see
# https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
class LoadableModel:
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
    def h5save (self, h5grp):
        h5grp.attrs['model_name'] = self.model_name
        for field in self.stored_fields:
            h5grp.create_dataset(field, data=self.__dict__[field])
        for attr in self.stored_attrs:
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
    def __init__ (self):
        self.M = 0
    def __len__ (self):
        return self.M
    # TODO should we define vector_dimension and matrix_dimension here??
    #@staticmethod
    #def load (h5data):
    #    newself = loaded_model()
    #    model_definition = h5data
    #    for attr,val in model_definition.attrs.items():
    #        newself.__dict__[attr] = val
    #    for field,val in model_definition.items():
    #        newself.__dict__[field] = np.array(val)
    #    return newself



def h5info (h5file):
    from . import __version__
    grp = h5file.create_group("mctpy")
    grp.attrs['version'] = __version__

def load_model (h5file):
    model_name = h5file.attrs.get('model_name','')
    # we just try the factory, it will raise an exception if the model
    # cannot be loaded, user needs to catch that
    # empty string (if model_name is missing) will load loaded_model
    return ModelFactory.get(model_name, h5file)

