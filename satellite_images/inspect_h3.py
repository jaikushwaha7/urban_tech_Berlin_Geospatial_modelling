import h3
import json
print(json.dumps([a for a in dir(h3) if not a.startswith('_')], indent=2))
