#!/usr/bin/env python3
"""
Ring Camera Authentication CLI Utility
Generates and saves the OAuth token for Ring Doorbell / Camera access.
"""

import json
import getpass
import sys
from pathlib import Path
from ring_doorbell import Auth

def token_updated(token):
    token_file = Path("ring_token.json")
    with open(token_file, "w", encoding="utf-8") as f:
        json.dump(token, f, indent=2)
    print(f"\n[+] OAuth token successfully saved to: {token_file.resolve()}")

def main():
    print("=" * 60)
    print("Ring Camera Authentication Setup")
    print("=" * 60)
    print("This utility securely connects to your Ring account and saves an")
    print("OAuth token to 'ring_token.json' so the rodent detector can access")
    print("the camera snapshot and battery levels.\n")

    username = input("Enter your Ring Email: ").strip()
    if not username:
        print("Error: Email is required.")
        sys.exit(1)

    password = getpass.getpass("Enter your Ring Password: ")
    if not password:
        print("Error: Password is required.")
        sys.exit(1)

    auth = Auth("RodentIdentification/1.0", token_updater=token_updated)

    try:
        auth.fetch_token(username, password)
        print("[+] Logged in successfully!")
    except Exception as e:
        # Check if 2FA code is needed
        print("\n[*] 2FA Authentication code required (SMS or Authenticator App).")
        code = input("Enter the 2FA Code received: ").strip()
        try:
            auth.fetch_token(username, password, otp=code)
            print("\n[+] 2FA Verified! Token stored in ring_token.json")
        except Exception as e2:
            print(f"\n[-] Authentication failed: {e2}")
            sys.exit(1)

    print("\n[SUCCESS] Ring setup complete. You can now launch the web app with:")
    print("  uv run python -m src.app")

if __name__ == "__main__":
    main()
