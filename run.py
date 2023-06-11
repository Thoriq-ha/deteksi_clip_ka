## Add all required imports here ##
## Eg. 
# import cv2
# import streamlit as st
# import pandas
##
# Python In-built packages
from pathlib import Path
import PIL


# External packages
import streamlit as st

# Local Modules
import settings
import helper

import os
from ultralytics import YOLO
import streamlit as st
import cv2
import pafy
import geocoder
from fpdf import FPDF
import base64
import pandas as pd
import settings
from datetime import datetime
import subprocess

if __name__ == '__main__':
    subprocess.run("streamlit run app.py")