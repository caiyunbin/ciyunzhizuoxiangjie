# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:33:24 2019

@author: Administrator
"""

from wordcloud import WordCloud
import cv2
import jieba
import jieba.posseg as psg

#读取一个数据集，取出内容，sum变成一个长的字符串
from datetime import datetime
import pandas as pd


def slice_data():
    #读取了数据将列名进行了修改
    data= pd.read_csv('C:/Users/Administrator/Desktop/文件集合/chuangku.csv',encoding='utf-8-sig',names=['汽车代号','ID','日期','帖子类型','内容'])
    #去除数据的重复行
    data = data.drop_duplicates()
    data['日期'] = pd.to_datetime(data['日期'])
    this_year = data[(data["日期"]>= datetime(2019,1,1)) & (data["日期"]<= datetime(2019,8,30))]
    return this_year
    
  
def year_apply(this_year):
    this_year['内容'] = this_year['内容'].astype(str)
    contents = this_year['内容'].sum()
    return contents

this_year = slice_data()
sentence = year_apply(this_year)

def get_stopword_list():
    stop_word_path = 'C:/Users/Administrator/Desktop/文件集合/stop_words.txt'
    stop_word_list = [article.replace('\n','') for article in open(stop_word_path,encoding = 'utf8').readlines()]
    return stop_word_list
   
#将这个字符串进行词性分词，取出名词和形容词
seg_list =psg.cut(sentence)
filter_list = []
stopword_list = get_stopword_list()
for seg in seg_list:
    word =seg.word
    flag =seg.flag
    if not flag.startswith('a'): #如果想要提取形容词和名词的话可以用这个代码if not (flag.startswith('n') or flag.startswith('n')):
        continue
    if not word in stopword_list and len(word)>1:
        filter_list.append(word)

##对词语列表的数量进行排序
def get_tf_dic(words,topik=100):
    tf_dic = {}
    for word in words:
        tf_dic[word]=tf_dic.get(word,0.0)+1.0
    for k,v in tf_dic.items():
        tf_dic[k] = int(v)
    return sorted(tf_dic.items(),key = lambda x:x[1],reverse = True)[:topik]

get_tf_dic(filter_list)    
#加载停用词词表，去掉没有用的词，取出长度大于2的词汇

text = '/n'.join(filter_list)

cut_text =" ".join(jieba.cut(text))
 
color_mask = cv2.imread('C:/Users/Administrator/Desktop/tf-idf documents/qiya.jpg')
 
cloud = WordCloud(
       #设置字体，不指定就会出现乱码
       font_path="C:/Users/Administrator/Desktop/tf-idf documents/STXINGKA.TTF",
       #font_path=path.join(d,'simsun.ttc'),
       #设置背景色
       background_color='white',
       #词云形状
       mask=color_mask,
       #允许最大词汇
       max_words=2000,
       #最大号字体
       max_font_size=60
   )
 
wCloud = cloud.generate(cut_text)
wCloud.to_file('C:/Users/Administrator/Desktop/文件集合/chuangku.jpg')
 
import matplotlib.pyplot as plt
plt.imshow(wCloud, interpolation='bilinear')
plt.axis('off')
plt.show()









