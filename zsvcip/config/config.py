import yaml
from easydict import EasyDict


class CfgNode(EasyDict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        else:
            d = dict(d)
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)

        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and not k in (
                "update",
                "pop",
                "update_from_config_file",
                "update_from_args",
            ):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, CfgNode):
            value = CfgNode(value)
        super(CfgNode, self).__setattr__(name.casefold(), value)
        super(CfgNode, self).__setitem__(name.casefold(), value)

    def __getattr__(self, name):
        for key, value in self.__dict__.items():
            if key.lower() == name.casefold():
                return value
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __getitem__(self, name):
        return self.__dict__[name.casefold()]

    def update_from_config_file(self, filename):
        self.update(CfgNode(yaml.load(open(filename, "r"), yaml.SafeLoader)))

    def update_from_args(self, args):
        opts = iter(args)
        for k, v in zip(opts, opts):
            keys = k.split(".")
            d = v
            for key in reversed(keys):
                d = {key: d}
            self.update(CfgNode(d))

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)

        for key, value in d.items():
            if key in self and isinstance(self[key], dict) and isinstance(value, dict):
                self[key].update(value)
            else:
                setattr(self, key, d[key])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(CfgNode, self).pop(k, d)
