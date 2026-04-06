import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from cvrewriteagent.agent import CVRewriteAgent
import gradio as gr
