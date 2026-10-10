"""Named DR02 model metadata shared by task configurations."""

import json
from pathlib import Path

_PROFILE_PATH = Path(__file__).parent / "description" / "profile.json"
PROFILE = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
JOINT_ORDER = tuple(PROFILE["joint_order"])
BODY_ORDER = tuple(PROFILE["body_order"])
TRACKING_BODY_NAMES = tuple(PROFILE["body_orders"]["tracking"])
AMP_MOTION_BODY_NAMES = tuple(PROFILE["body_orders"]["amp_motion"])
TRACKING_ANCHOR_BODY = PROFILE["tracking_anchor_body"]
