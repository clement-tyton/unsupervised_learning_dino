"""
Check the S3 session creds in .env: when does the token expire, does S3 accept it now?

The tytonai S3 creds (AWS_SESSION_TOKEN) are a short-lived JWT. This decodes its `exp` claim
to show minutes-left, then pings S3 to confirm. The AccessDenied `"exp" claim timestamp check
failed` error just means the token expired — refresh the AWS_* values in .env and re-run.

Run it line by line (Shift+Enter) or: .venv/bin/python check_s3.py
"""

import base64
import json
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv


def token_seconds_left(token: str) -> float | None:
    """Seconds until a JWT's `exp` (negative = already expired), or None if not a JWT/exp."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1] + "=" * (-len(parts[1]) % 4)              # pad base64url
    exp = json.loads(base64.urlsafe_b64decode(payload)).get("exp")
    return None if exp is None else exp - time.time()


def s3_reachable() -> tuple[bool, str]:
    """(ok, message): try a 1-object list on the file bucket — proves creds + endpoint work."""
    from tytonai_utils.s3 import make_s3_client
    try:
        make_s3_client().list_objects_v2(Bucket=os.environ["S3_FILE_BUCKET"], MaxKeys=1)
        return True, "S3 accepted the creds"
    except Exception as e:                                       # botocore ClientError, etc.
        return False, f"{type(e).__name__}: {e}"


def check_s3() -> float | None:
    """Reload .env, report token expiry + live S3 connectivity. Returns seconds left (or None)."""
    load_dotenv(".env", override=True)                          # re-export fresh creds into this process
    left = token_seconds_left(os.environ.get("AWS_SESSION_TOKEN", ""))
    if left is None:
        print("AWS_SESSION_TOKEN: not a decodable JWT (can't read exp)")
    else:
        when = datetime.fromtimestamp(time.time() + left, timezone.utc)
        state = "EXPIRED" if left <= 0 else f"valid {left / 60:.0f} min"
        print(f"token exp: {when:%Y-%m-%d %H:%M} UTC  ->  {state}")
    ok, msg = s3_reachable()
    print(f"S3 ping  : {'OK' if ok else 'FAILED'} — {msg}")
    if not ok:
        print("  -> refresh AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN in .env, then re-run")
    return left


# ════════════════════════════════════════════════════════════════════════════════════
#  RUN
# ════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    check_s3()
