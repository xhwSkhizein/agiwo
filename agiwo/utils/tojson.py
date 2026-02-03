import orjson
from dataclasses import asdict


def fill_none_converter(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {k: v for k, v in asdict(obj).items() if v is not None}
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def to_json(obj) -> str:
    return orjson.dumps(obj, default=fill_none_converter).decode("utf-8")
