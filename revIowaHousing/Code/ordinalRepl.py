 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:11:09 2020
@author: Matt Davis, Ragy, Alex, Stephen
 Takes series / dict / int 
 Fill na's with recieved int value
 return series with updated dictionary
 """

def ordinalRepl(pds, ordinalDict, fill):
    ordinalSeries = pds.replace(ordinalDict)
    ordinalSeries = ordinalSeries.fillna(fill)
    return ordinalSeries
