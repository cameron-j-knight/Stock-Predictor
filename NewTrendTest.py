import pytrends.request as request
import base64
"""
Author: Cameron Knight
Copyright 2017, Cameron Knight, All rights reserved.
"""

userName = "camknightt@gmail.com"
password = "L1laiteigo!go"
userName = "jeffeled@gmail.com"
password = "Avpr1423"

trend_request = request.TrendReq(userName, password, "asdasd")
trend_request.build_payload(['Hey'])