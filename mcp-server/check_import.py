import importlib.util
from pathlib import Path

p = Path(__file__).parent / 'master_orchestrator_api.py'
spec = importlib.util.spec_from_file_location('mo', str(p))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('APP_PRESENT', hasattr(m, 'app'))

