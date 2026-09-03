import asyncio
import json
import sys
from pathlib import Path
from ring_doorbell import Auth

def token_updated(token):
    token_file = Path("ring_token.json")
    with open(token_file, "w", encoding="utf-8") as f:
        json.dump(token, f, indent=2)
    print("TOKEN_SAVED_SUCCESS")

async def authenticate(otp: str):
    auth = Auth("RodentIdentification/1.0", token_updater=token_updated)
    try:
        token = await auth.async_fetch_token("SeanBaker513@gmail.com", "AlbieHerbie1!", otp_code=otp)
        token_updated(token)
        print("2FA_LOGIN_SUCCESS")
    except Exception as e:
        print(f"2FA_LOGIN_ERROR: {type(e).__name__} - {e}")
        sys.exit(1)
    finally:
        await auth.async_close()

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python src/complete_ring_2fa.py <OTP_CODE>")
        sys.exit(1)

    otp = sys.argv[1].strip()
    asyncio.run(authenticate(otp))

if __name__ == "__main__":
    main()
