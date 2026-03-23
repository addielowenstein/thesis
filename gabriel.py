import gabriel
from gabriel.utils.plot_utils import regression_plot, bar_plot, box_plot, line_plot
import os
import pandas as pd

import json
from pathlib import Path

def main():

    with open(Path("api_keys.json"), "r") as f:
        api_keys = json.load(f)

    openai_api_key = api_keys["openai_api_key"]

    


if __name__ == "__main__":
    main()


