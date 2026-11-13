"""Version 2 orchestration: full-book MC, SA-CCR v2 and CSA analytics."""

from . import analytics as analytics_mod
from . import fullbook as fullbook_mod
from . import saccr_v2 as saccr_v2_mod


def run_v2(bundle):
    fullbook = fullbook_mod.run_full_book(bundle["live_book"])
    saccr = saccr_v2_mod.build_saccr_v2(bundle, fullbook)
    analytics = analytics_mod.run_csa_analytics(bundle)
    return {"fullbook": fullbook, "saccr_v2": saccr, "analytics": analytics}
