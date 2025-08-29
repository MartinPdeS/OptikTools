from pydantic.dataclasses import ConfigDict


config_dict = ConfigDict(
    arbitrary_types_allowed=True, kw_only=True, slots=True, extra="forbid"
)
