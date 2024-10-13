import os
from setuptools import setup, find_packages
from setuptools.command.install import install

setup(
    package_dir={"": "src"},  # This points setuptools to the src directory
    packages=["poliprompt"],  # This is your main package in src/poliprompt
)


# DEFAULT_WORKING_STATION = os.path.expanduser("~/.cache/")


# def create_working_station(install):
#     """
#     Creates a working station directory and saves the default config.yaml file.
#     """
#     def run(self):
#         print("\n---- PoliPrompt Installation ----")
#         install.run(self)

#         # Workspace paths
#         default_cache_dir = os.path.join(DEFAULT_WORKING_STATION, "poliprompt")
#         cache_dir = os.path.expanduser(default_cache_dir)
#         config_dir = os.path.expanduser(os.path.join(default_cache_dir, "configs"))
#         config_file = os.path.join(config_dir, "config.yaml")

#         # Create directories if they don't exist
#         os.makedirs(cache_dir, exist_ok=True)
#         os.makedirs(config_dir, exist_ok=True)

#         # Write the default configuration file
#         if not os.path.exists(config_file):
#             with open(config_file, "w") as f:
#                 f.write(
#                     """# Default PoliPrompt configuration workspace: ~/.cache/poliprompt/log_level: INFO # Add other default configurations as needed"""
#                 )
#         print(f"Workspace created at {cache_dir}")
#         print(f"Configuration file created at {config_file}")

# setup(
#     cmdclass={
#         'install': CustomInstallCommand,  # Use the custom installation class
#     },
#     package_dir={"": "src"},  # This points setuptools to the src directory
#     packages=["poliprompt"],  # This is your main package in src/poliprompt
# )
