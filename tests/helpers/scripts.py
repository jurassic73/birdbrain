"""Shared helpers for the scripts tests (used by more than one test module)."""

PASS = "p@ss!$;#x"


FAKE = f"""# comment

GROUND_ZONE=0.1,0.2;0.3,0.4;0.5,0.6
export CAM_HOST=10.0.0.9
RTSP_USER='bob'
RTSP_PASS="{PASS}"
RTSP_PORT=8554
"""
