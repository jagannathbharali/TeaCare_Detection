import sys
from pathlib import Path
 
# Ensure project root is on the path so all imports resolve
sys.path.insert(0, str(Path(__file__).parent))
 
from ui.main_window import launch
 
if __name__ == "__main__":
    launch()
