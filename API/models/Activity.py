# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:51:19 2020

@author: win10
"""
from pickle import LIST
from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class Activity_Data(BaseModel):
    X1:list=[]
    Y1:list=[]
    Z1:list=[]
    X2:list=[]
    Y2:list=[]
    Z2:list=[]
    