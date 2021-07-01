from ruamel.yaml import YAML
from collections import namedtuple, OrderedDict
from recordclass import recordclass
from typing import Dict
import pprint

def create_namedtuple_from_dict(
    obj: Dict, meta: Dict = None, maxdepth=-1, depth=0
):
    #  we pass a python dict already but want a namedtuple for convenience
    if maxdepth != -1 and depth >= maxdepth:
        return obj
    if isinstance(obj, dict):
        if depth == 0 and meta is not None:
            obj["meta"] = meta
        fields = sorted(obj.keys())
        #  instead of normal namedtuple we use recordclass for now
        namedtuple_type = recordclass(
            typename="GenericObject", fields=fields, rename=True,
        )
        field_value_pairs = OrderedDict(
            (
                str(field),
                create_namedtuple_from_dict(
                    obj[field], maxdepth=maxdepth, depth=depth + 1
                )
                if field not in ["args"]
                else obj[field],
            )
            for field in fields
        )
        try:
            return namedtuple_type(**field_value_pairs)
        except TypeError:
            # Cant create namedtuple instance,
            # fallback to dict (invalid attribute names)
            return dict(**field_value_pairs)
    elif isinstance(obj, (list, set, tuple, frozenset)):
        return [
            create_namedtuple_from_dict(
                item, maxdepth=maxdepth, depth=depth + 1
            )
            for item in obj
        ]
    else:
        return obj


def convert_config(config_str: str, filename: str, collection_type=None):
    parser = YAML(typ="safe")
    config = parser.load(config_str)  # dict representation of yaml file
    return (
        create_namedtuple_from_dict(
            config, meta={"config_str": config_str, "filename": filename}
        ),
        config,
    )


def load_config(filename: str):
    with open(filename, "r") as f:
        config_str = f.read()  # string representation of yaml file
    # return convert_config(config_str, filename)
    cfg_named_tuple, cfg_dict = convert_config(config_str, filename)
    return cfg_named_tuple, config_str, cfg_dict


def save_config(config_data, filename):
    with open(filename, "w") as f:
        f.write(config_data.meta.config_str)


if __name__ == "__main__":
    path = "/home/ditzel/radar/confi.yaml"
    config = load_config(path)
    # print(config)
    # print(type(config))
    # print(config.general.experiment_name)
