import traceback
try:
    from openwakeword.model import Model
    print("OK")
except Exception:
    traceback.print_exc()
