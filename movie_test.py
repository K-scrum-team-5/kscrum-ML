import pandas as pd
import numpy as np
import math

import warnings
warnings.filterwarnings(action='ignore')


# ---------------- 영화 데이터 불러오기 ----------------
movie=pd.read_csv("./csv/movie.csv")
print(movie.head())

