# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:22:16 2020

@author: User
"""
import pandas as pd
import numpy as np
import re
holiday=pd.read_csv('holiday.csv',encoding='utf-8',names=['date','is_holiday'])
data=pd.read_csv('sample.csv',encoding='utf-8')
weather=pd.read_csv('weather2.csv',encoding='utf-8')
cpi=pd.read_csv('cpi.csv',encoding='utf-8')
#change / to -
holiday['date']=holiday['date'].astype(str).apply(lambda x: x.replace("/", "-"))
holiday['date']=pd.to_datetime(holiday['date'])
holiday['date'] = [d.date() for d in holiday['date']]
holiday['date'] = holiday['date'].astype(str)


data['inv_time']=pd.to_datetime(data['inv_time'])
data['date'] = [d.date() for d in data['inv_time']]
data['time'] = [d.time() for d in data['inv_time']]
data['date']=data['date'].astype(str)
real_data=data[['inv_time','date', 'time','seller_address','inv_id']]
real_data=data

#merge holidays and real_data
#''''address processing'''
#遺失值以max填入
real_data['seller_address']=real_data['seller_address'].fillna('台北市中山區松江路183號183-1號1樓')

#spliting 
def county(x,num):
    if len(str(x).split('縣'))>=num:
        return x.split('縣',1)
    elif len(str(x).split('市'))>=num:
        return x.split('市',1) 
    else :
        return x

    

real_data['county']=real_data['seller_address'].apply(county,args=(2,))
real_data[['county','area']]=pd.DataFrame(real_data['county'].tolist()).iloc[:,:2]#新增欄位 country area
real_data['county']=real_data['county'].apply(county,args=(2,))
road=real_data[real_data['county'].apply(lambda x: isinstance(x, list))]#road :找切的不乾淨的row(ex:心北市板橋區縣民大道)
road=road.reset_index()
total_road=road[['index','area']]
r=pd.DataFrame(road['county'].tolist())
r.index=road['index']
#切乾淨後回填
for i in r.index:
    real_data['county'][i]=r[0][i]
    real_data['area'][i]=r[1][i]
    
  
#修正前面有地址or臺北(非台北)的地址
real_data['county']=real_data['county'].apply(lambda x: x.replace('地址:',''))
real_data['county']=real_data['county'].apply(lambda x: x.replace('臺','台'))
real_data['county']=real_data['county'].apply(lambda x: x.replace('台灣',''))
real_data['county']=real_data['county'].apply(lambda x: x.replace(' ',''))


#修正ex:300新竹市有郵遞區號的row
def del_post_number(x):
    a=re.findall(r'\d+', x)
    if len(a)>=1:
        for i in a:
            x=x.replace(i,'')
    return x

real_data['county']=real_data['county'].apply(del_post_number)


#修正 ex:北市or竹縣縮寫的地址
def change_abbreviate(real_data,word,replace_word):
    a=pd.Series(real_data[real_data['county']==word]['county'].index)
    for d in a:
        real_data['county'][d]=replace_word       
word='北'
replace_word='台北'
change_abbreviate(real_data, word, replace_word)


#切開area ex:板橋區 湖口鄉....ok是避免切兩次
def area(x,num,ok):      
    if len(str(x).split('區'))>=num:
        ok=1
        return x.split('區',1)
    elif (len(str(x).split('鎮'))>=num)&(ok==0):
        ok=1
        return x.split('鎮',1) 
    elif (len(str(x).split('市'))>=num)&(ok==0):
        ok=1
        return x.split('市',1) 
    elif (len(str(x).split('鄉'))>=num)&(ok==0):
        ok=1
        return x.split('鄉',1) 
    else :
        return x

real_data['area2']=real_data['area'].apply(area,args=(2,0))
real_data[['area3','area4']]=pd.DataFrame(real_data['area2'].tolist()).iloc[:,:2]
#沒切乾淨者回填
for i,index in zip(total_road.index,total_road['index']):
    real_data['area4'][index]=total_road['area'][i]
#沒有行政單位者ex:台北市光復南路(沒有大安區)
#area2沒有成功被切分成list代表裡面沒有行政區
road=real_data[real_data['area2'].apply(lambda x: isinstance(x, str))]
query=pd.DataFrame(road['seller_address'].unique(),columns=['seller_address'])
query['area3']=0
query['road']=0
API_KEY='AIzaSyAzxVWS5a16m_MvaYiyYZpDfJc-lrA4qYM'
import googlemaps
from google.cloud import translate_v2 as translate
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'trans.json'
a=translate.Client()
gmaps=  googlemaps.Client(key=API_KEY)
#google geocode api 查詢完整地址取出行政區再翻譯成中文之後回填用cloud translate api
for index,address in zip(query.index,query['seller_address']):
#text='Zhongzheng District'
    ad=gmaps.geocode(address) 
    ad=ad[0]
    place=ad['formatted_address'].split(', ')[-3]
    #rp=ad['formatted_address'].split(', ')[-4]
    output=a.translate(place,source_language='en',target_language='zh_tw')
    #output_road=a.translate(rp,source_language='en',target_language='zh_tw')
    query['area3'][index]=output['translatedText']
    #query['road'][index]=output_road['translatedText']

    
query['area3']=query['area3'].apply(lambda x: x.replace('區',''))
test=real_data
test=test.reset_index()
test=test.merge(query,on=['seller_address'])

#處理新竹香山和台北象山音譯一樣
i=test[(test['county']=='新竹')&( test['area3_y']=='象山')]['area3_y'].index
for index in list(i):
    test['area3_y'][index]='香山'
#回填
for index,area in zip(test['index'],test['area3_y']):
    real_data['area3'][index]=area
    


real_data=real_data.drop(columns=['area','area2'])

for i,index in zip(test.index,test['index']):
    real_data['area4'][index]=test['area2'][i]

##road

def road(x,num,ok):      
    if len(str(x).split('里'))>=num:
        ok=1
        return x.split('里',1)
    elif (len(str(x).split('街'))>=num)&(ok==0):
        ok=1
        return x.split('街',1) 
    elif (len(str(x).split('道'))>=num)&(ok==0):
        ok=1
        return x.split('道',1) 
    elif (len(str(x).split('村'))>=num)&(ok==0):
        ok=1
        return x.split('村',1) 
    elif (len(str(x).split('路'))>=num)&(ok==0):
        ok=1
        return x.split('路',1) 
    
    
    
real_data['area5']=real_data['area4'].apply(road,args=(2,0))
#nonetype
real_data['area5']=real_data['area5'].fillna(0)
real_data=real_data.reset_index()
r=pd.DataFrame(real_data[real_data['area5']==0]['seller_address'].unique(),columns=['seller_address'])
r['area5']=0
for index,address in zip(r.index,r['seller_address']):
#text='Zhongzheng District'
    ad=gmaps.geocode(address) 
    ad=ad[0]
    place=ad['address_components'][-5]['long_name']
    #rp=ad['formatted_address'].split(', ')[-4]
    output=a.translate(place,source_language='en',target_language='zh_tw')
    #output_road=a.translate(rp,source_language='en',target_language='zh_tw')
    r['area5'][index]=output['translatedText']
    #query['road'][index]=output_road['translatedText']


r['area5']=r['area5'].fillna(0).astype(str)
r['area5']=r['area5'].apply(lambda x: x.replace('路',''))
r['area5']=r['area5'].apply(lambda x: x.replace('街',''))
r['area5']=r['area5'].apply(lambda x: x.replace('道',''))
r['area5']=r['area5'].apply(lambda x: x.replace('里',''))
r['area5']=r['area5'].apply(lambda x: x.replace('村',''))


def get_road(x):
    if(type(x)==list):
        a=x[0]
        return a
    else:
        return x   
real_data['area5']=real_data['area5'].apply(get_road)

for word,address in zip(r['seller_address'],r['area5']):
    contain=real_data['seller_address'].str.contains(word, regex=False).reset_index()
    contain=contain[contain['seller_address']==True]
    for i in contain['index']:
        real_data['area5'][i]=address







'''經費不足(得到location type)
#location type
location=pd.DataFrame(real_data['seller_address'].unique(),columns=['seller_address'])
location['location_type']=0 

for index,address in zip(location.index,location['seller_address']):
    try:
        ad=gmaps.geocode(address) 
        ad=ad[0]['geometry']['location']
        a=gmaps.places_nearby(location='25.0436529,121.5323979',radius=1)
        location['location_type'][index]=ad['geometry']['location_type']
    
    except:
        pass

'''


#''''weather 得到每日該行政區的平均溫度和平均雨量

weather=weather[['縣市','鄉鎮市區','年份', '月份', '日期','溫度','降雨量','日照時數']]
weather[['縣市','鄉鎮市區','年份', '月份', '日期']]=weather[['縣市','鄉鎮市區','年份', '月份', '日期']].astype(str)


weather['縣市']=weather['縣市'].apply(lambda x: x.replace('市',''))
weather['縣市']=weather['縣市'].apply(lambda x: x.replace('縣',''))
weather['縣市']=weather['縣市'].apply(lambda x: x.replace('臺','台'))
weather['鄉鎮市區']=weather['鄉鎮市區'].apply(lambda x: x.replace('鄉',''))
weather['鄉鎮市區']=weather['鄉鎮市區'].apply(lambda x: x.replace('鎮',''))
weather['鄉鎮市區']=weather['鄉鎮市區'].apply(lambda x: x.replace('市',''))
weather['鄉鎮市區']=weather['鄉鎮市區'].apply(lambda x: x.replace('區',''))
weather['group_id']=weather['縣市'].astype(str).str.cat(weather[['鄉鎮市區','年份', '月份', '日期']],sep=",")
#去除-9999
columns=['溫度','降雨量','日照時數']

for col in columns:
    a=weather[weather[col]<-1][col].index
    for index in a:
        weather[col][index]=None
    weather[col]=weather[col].fillna(weather[col].mean())
weather['日照時數']=weather['日照時數'].round(2)
w=weather.groupby('group_id')
wdata=w.agg(np.mean)
wdata=wdata.reset_index()

#concat weather and real_data

real_data['group_id']=real_data['date'].apply(lambda x: x.replace('-',','))
real_data['group_id']=real_data['county'].astype(str).str.cat(real_data[['area3','group_id']],sep=',')

#concat cpi
cpi=cpi.dropna(axis=0)
cpi=cpi.rename(columns={" ": "date"})
real_data['cpi_id']=pd.to_datetime(real_data['date'])
real_data['y']=pd.DatetimeIndex(real_data['cpi_id']).year
real_data['m']=pd.DatetimeIndex(real_data['cpi_id']).month.map("{:02}".format)
real_data['cpi_id']=real_data['y'].astype(str).str.cat(real_data['m'].astype(str),sep='-')

#還未merge




