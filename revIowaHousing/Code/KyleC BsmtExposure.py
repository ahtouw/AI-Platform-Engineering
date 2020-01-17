#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kyle Storm Cloud

This is my contribution to the transformation of ordinal features
into continuous features.
"""

# This dictionary maps the ordinal values in BsmtExposure to continuous values.
BsmtExposure_dict = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No':1, 'NA':0}

# A null value in BsmtExposure corresponds with "no basement"
# Denote the lack of a basement with 0
BsmtExposure_fillval = 0

