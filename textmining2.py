# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 13:02:47 2020

@author: User
"""

import pandas as pd
import numpy as np

from gensim.models import Word2Vec
import nltk
from nltk.cluster import KMeansClusterer
import jieba

  
from sklearn import cluster
from sklearn import metrics
from sklearn.manifold import TSNE
import re,datetime


''' product text preprocessing '''

#established stop words
stopWords = []
jieba.set_dictionary('dict.txt')

with open('stopWords.txt', 'r', encoding='UTF-8') as file:   #建立停用詞列表
    for data in file.readlines():
        data = data.strip()
        stopWords.append(data)
        
#read data and get text data to array      
data=pd.read_csv('sample.csv',encoding='UTF-8')
text=np.array(data['product_name']).tolist()

#part of speech tagging
import jieba.posseg as pseg
#分成兩部分儲存:training_set:保留名詞 totalset:存放原本斷詞後結果(以免篩選出來一個商品連一個名詞也沒有)
train=[]
training_set=[]
tr=[]
total_set=[]
for t in text:
    a = re.findall('[\u4e00-\u9fa5a-zA-Z]+',t,re.S)   #只要字符串中的中文，字母
    a = "".join(a)
    print(a)
    seg_list = jieba.lcut(a, cut_all=False)
    remainderWords=pseg.cut(a,seg_list)
    for word,flag in remainderWords:
        print('%s, %s' % (word, flag))
        if flag =='N':
            train.append(word)
        tr.append(word)                
    training_set.append(train)
    total_set.append(tr)
    train=[]
    tr=[]
    


train=[]
add=[]
i=0 
#回填篩出來一個名詞也沒有的training set  
for order in training_set:
    if len(order)==0:
        print(i)
        training_set[i]=total_set[i]
    i=i+1
#原data之缺失值(之後改成以該年齡層最多購買次數產品進行回填)
for order in training_set:     
    if len(order)==0:
        order.append('0')

        
#進行text mining

model = Word2Vec(training_set, min_count=1)
#建立sentence vector
def sent_vectorizer(sent, model):  
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
     
    return np.asarray(sent_vec) / numw
  
  
X=[]
for record in training_set:
    X.append(sent_vectorizer(record, model))   
 
print ("========================")
print(X) #印出每個標題的vector

#找到分幾群才能找到群心
i=15
while i>0:
    NUM_CLUSTERS=i
    try:
        kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
        assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
        print('k=',i,' success')
        break
    except:
        print('k=',i,'failed')
        i=i-1
     
     
kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)
  
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
#plotting#
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE
 
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
 
Y=model.fit_transform(X)
 
plt.scatter(Y[:, 0], Y[:, 1], c=assigned_clusters, s=290,alpha=.5)
 
for j in range(len(text)):    
   plt.annotate(assigned_clusters[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
   print ("%s %s" % (assigned_clusters[j],  text[j]))
plt.show() 

#transfer list to dataframe and merge back to origin data
df_clusters=pd.DataFrame(assigned_clusters,columns=['cluster'])
df_product=pd.DataFrame(text,columns=['product'])
df_p=pd.concat([df_product,df_clusters],axis=1)
data2=data.merge(df_p,left_index=True, right_index=True)



'''date concat with id and grouping'''
data2['inv_time']=pd.to_datetime(data2['inv_time'])
data2['date'] = [d.date() for d in data2['inv_time']]
data2['time'] = [d.time() for d in data2['inv_time']]
data2['date']=data2['date'].astype(str)


''''address processing'''
#遺失值以max填入
data2['seller_address']=data2['seller_address'].fillna('台北市中山區松江路183號183-1號1樓')
#spliting 
def county(x,num):
    if len(str(x).split('縣'))>=num:
        return x.split('縣',1)
    elif len(str(x).split('市'))>=num:
        return x.split('市',1) 
    else :
        return x
    

data2['county']=data2['seller_address'].apply(county,args=(2,))
data2[['county','area']]=pd.DataFrame(data2['county'].tolist())#新增欄位 country area
data2['county']=data2['county'].apply(county,args=(2,))
road=data2[data2['county'].apply(lambda x: isinstance(x, list))]#road :找切的不乾淨的row(ex:心北市板橋區縣民大道)
r=pd.DataFrame(road['county'].tolist())
r.index=road.index
#切乾淨後回填
for i in r.index:
    data2['county'][i]=r[0][i]
    data2['area'][i]=r[1][i]

'''this person buy how many product that belongs to the cluster at this day'''
#建立clusters 欄位(clusters是將來要帶入模型的資料)
data2['group_id']=data2['inv_id'].astype(str).str.cat(data2['date'],sep=",")
d_id=data2['group_id'].unique()
d=data2.groupby('group_id')
clusters=pd.DataFrame(data2['cluster'].unique()).T#clusters schema
clusters=clusters.drop(clusters.index[0])
clusters['date']=0
clusters['inv_id']=0
clusters['daily_spend']=0
clusters['age']=0
clusters['gender']=0
clusters['county']=0
clusters['area']=0
clusters['group_id']=0
#......



#將分群結果填入schema
for i in d_id:
       try: 
           test=d.get_group(i)
           a=test['cluster'].value_counts()
           insert_data=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,i]
           for i in a.index:
               insert_data[i]=1
               
           insert_df=pd.DataFrame(np.array(insert_data).reshape(1,len(insert_data)),columns=clusters.columns)
           clusters = pd.concat([clusters,insert_df],axis=0, ignore_index=True)
           print('success')
       except:
           print('no match')
           
#回填日期和使用者 (包含在group_id)
clusters['date']=clusters['group_id'].apply(lambda x: x.split(',')[1])
clusters['inv_id']=clusters['group_id'].apply(lambda x: x.split(',')[0])
clusters['age']=clusters['inv_id'].apply(lambda x:data2[data2['inv_id']==int(x)]['age'].to_list()[0])
clusters['gender']=clusters['inv_id'].apply(lambda x:data2[data2['inv_id']==int(x)]['gender'].to_list()[0])
clusters['county']=clusters['inv_id'].apply(lambda x:data2[data2['inv_id']==int(x)]['county'].to_list()[0])
clusters['area']=clusters['inv_id'].apply(lambda x:data2[data2['inv_id']==int(x)]['area'].to_list()[0])
clusters['daily_spend']=clusters['group_id'].apply(lambda x:data2[data2['group_id']==x]['amonut'].sum())
clusters.sort_values(by=['group_id'])
#這一步之後有改 SAMPLE_DATA
#計算下及上次購買
def next_day(group_id):
    try:
        inv_id=group_id.split(',')[0]
        date=group_id.split(',')[1]
        x=clusters[clusters['inv_id']==inv_id].reset_index() #單一使用者的資料表
        x=x.sort_values(by=['date']).reset_index()
        ans=x[x['group_id']==group_id].index[0]
        return x['date'][ans+1]
        
    except:
        return None    
        
def previous_day(group_id):
    try:
        inv_id=group_id.split(',')[0]
        date=group_id.split(',')[1]
        x=clusters[clusters['inv_id']==inv_id].reset_index() #單一使用者的資料表
        x=x.sort_values(by=['date']).reset_index()
        ans=x[x['group_id']==group_id].index[0]
        return x['date'][ans-1]       
    except:
        return None 
       
clusters['next_date']=clusters['group_id'].apply(next_day)
clusters['pre_date']=clusters['group_id'].apply(previous_day)
#計算相隔天數
clusters['date']=clusters['date'].apply(lambda x: datetime.date.fromisoformat(x))
clusters['next_date']=clusters['next_date'].apply(lambda x: datetime.date.fromisoformat(x) if x != None else None)
clusters['pre_date']=clusters['pre_date'].apply(lambda x: datetime.date.fromisoformat(x) if x != None else None)
def num_nday(df):
    try:
        interval=df['next_date']-df['date']
        return interval.days
    except:
        return None

def num_pday(df):
    try:
        interval=df['date']-df['pre_date']
        return interval.days
    except:
        return None

clusters['num_ndate']=clusters.apply(num_nday,axis=1)
clusters['num_pdate']=clusters.apply(num_pday,axis=1)
#塑膠袋
def plastic(group_id):
    x=data2[data2['group_id']==group_id].reset_index(drop=True)
    y=x.product_name.str.contains('袋', regex=False)
    y=y[y].index
    if  len(y) != 0:
        index_plastic=y[0]
        if int(x['unit_price'][index_plastic]) < 10:
            return 1
        else: return 0
    else: return 0
clusters['plastic']=clusters['group_id'].apply(plastic)
#RMF_F
clusters['RMF_F']=clusters['group_id'].apply(lambda x:len(data2[(data2['group_id']=='23066,2019-02-27') & (data2['item_no']==1)]))



clusters=clusters['num_ndate'].fillna(-1)

c=clusters[clusters['num_ndate']!=-1]