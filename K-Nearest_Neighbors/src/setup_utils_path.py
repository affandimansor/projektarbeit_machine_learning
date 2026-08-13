import os
import sys

# Navigate 2 levels up from 'src' to 'Projektarbeit_Machine_Learning'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))

if project_root not in sys.path:
    sys.path.append(project_root)