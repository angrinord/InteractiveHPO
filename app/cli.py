import importlib.util
import sys

def main():
    from streamlit.web import cli as stcli
    script = importlib.util.find_spec("run").origin
    sys.argv = ["streamlit", "run", script]
    sys.exit(stcli.main())
