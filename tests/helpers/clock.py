"""The frozen clock every test runs on. conftest patches `datetime`/`date` in every birdbrain module with these."""
from datetime import date as _real_date, datetime as _real_datetime

# "Now" for every test: midday on the day the run started, so a run that crosses midnight (or starts just past it)
# sees one fixed day. Tests that need a time relative to now use the `now` fixture or import FROZEN_NOW from here.
FROZEN_NOW = _real_datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)


class _Meta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, cls.__mro__[1])


class FrozenDatetime(_real_datetime, metaclass=_Meta):
    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return FROZEN_NOW
        return FROZEN_NOW.astimezone().astimezone(tz)

    @classmethod
    def today(cls):
        return FROZEN_NOW


class FrozenDate(_real_date, metaclass=_Meta):
    @classmethod
    def today(cls):
        return FROZEN_NOW.date()
