import pandas as pd
import numpy as np
import json
from bs4 import BeautifulSoup
import re
from langdetect import detect
import text2text as t2t
import requests
import hashlib
import random

# APP ID
appID = '20211109000995394'
# 密钥
secretKey = 'b9xwuX_bCSwxaz0EWmG3'
# 百度翻译 API 的 HTTP 接口
apiURL = 'https://fanyi-api.baidu.com/api/trans/vip/translate'

def gettext(list):
    """
    将list数据转换成text字符串，去除非文本数据，去除html标签
    :param list: list
    :return: str
    """
    tokemodel = BeautifulSoup(list)
    return tokemodel.get_text()


def text_process(text):
    """
    去除标点符号，数字
    :param text: str
    :return: str
    """
    letters_only = re.sub("[^a-zA-Z]",  # The pattern to search for
                          " ",  # The pattern to replace it with
                          text)  # The text to search
    return letters_only

def getwords(text):
    """
    将处理好的（去除标点的文本）转化为词列表，将字母大小写改进，全部改为小写
    :param text: str
    :return: list
    """
    text = text.lower()
    words = text.split()
    return words

def lang_detect(text):
    """
    检测语言类型，返回str
    :param text: str
    :return: 语言类型 str
    """
    return detect(text)

def translate(text):
    """
    传送str数据，返回翻译后的str
    :param text: str
    :return: str
    """
    t2t.Transformer.PRETRAINED_TRANSLATOR = "facebook/m2m100_418M"
    lines = text.split('\n')
    from_lang = detect(lines[0])
    tgt_text = ''
    for line in lines:
        tgt_text = tgt_text + t2t.Handler(line, src_lang=from_lang).translate(tgt_lang='en')[0]
    return tgt_text










