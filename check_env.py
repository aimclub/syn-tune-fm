# check_env.py
import sys

def check_import(module_name):
    try:
        __import__(module_name)
        print(f"✅ {module_name} imported successfully")
    except ImportError as e:
        print(f"❌ {module_name} failed: {e}")

print(f"Python version: {sys.version}")
print("-" * 20)
check_import("hydra")
check_import("torch")
check_import("pandas")
check_import("tabpfn")
check_import("tabtune") # Важно проверить, встал ли он через git