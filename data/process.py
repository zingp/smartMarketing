#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
解析原始数据：'../files/服饰_50k.json'
转换成样本需要的格式： 语料与参考数据用<sep>分割
划分训练集、验证集、测试集
"""
import os
import sys
import json
import jieba
import pathlib
abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))
from data_utils import write_samples, partition

print("abs_path:", abs_path)

samples = set()
# Read json file.
json_path = os.path.join(abs_path, '../files/服饰_50k.json')
with open(json_path, 'r', encoding='utf8') as file:
    jsf = json.load(file)

for jsobj in jsf.values():
    title = jsobj['title'] + ' '  # Get title.
    kb = dict(jsobj['kb']).items()  # Get attributes.
    kb_merged = ''
    for key, val in kb:
        kb_merged += key+' '+val+' '  # Merge attributes.

    ocr = ' '.join(list(jieba.cut(jsobj['ocr'])))  # Get OCR text.
    texts = []
    texts.append(title + ocr + kb_merged)  # Merge them.
    reference = ' '.join(list(jieba.cut(jsobj['reference'])))
    for text in texts:
        sample = text + '<sep>' + reference  # Seperate source and reference.
        samples.add(sample)
write_path = os.path.join(abs_path, '../files/samples.txt')
write_samples(samples, write_path)
partition(samples)
