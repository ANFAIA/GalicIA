import os, sys
import streamlit.web.cli as stcli

if __name__ == "__main__":
    sys.argv = ["CUDA_VISIBLE_DEVICES=\"-1\"","streamlit", "run", "app.py"]
    sys.exit(stcli.main())