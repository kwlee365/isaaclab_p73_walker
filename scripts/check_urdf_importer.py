"""List all available URDF import config options."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import omni.kit.commands
_, ic = omni.kit.commands.execute("URDFCreateImportConfig")
methods = [m for m in dir(ic) if "set_" in m]
print("\n" + "=" * 60)
print("Available URDF ImportConfig set_ methods:")
print("=" * 60)
for m in sorted(methods):
    print(f"  {m}")
print("=" * 60)

simulation_app.close()
