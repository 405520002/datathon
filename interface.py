# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 18:30:17 2020

@author: User
"""
import pandas as pd
import numpy as np
data=pd.read_csv('')



def get_values():
    x=num01.get()
    x=int(x)
    x2=int(num02.get())
    result01.set(str(x+x2))
    


from tkinter import *
window01=Tk()
window01['bg']='light yellow'
window01.geometry("2000x1000")
user_id=StringVar()
num=IntVar()
next_time=StringVar()
next_place=StringVar()
today_date=StringVar()
product=StringVar()
Label(window01,width=10,text="輸入user_id:",bg='light yellow',font=('microsoft yahei', 16, 'bold')).grid(row=0,column=0)
Label(window01,width=10,text="今日購買商品",bg='light yellow',font=('microsoft yahei', 16, 'bold')).grid(row=0,column=3)
Label(window01,width=10,text="今日購買日期",bg='light yellow',font=('microsoft yahei', 16, 'bold')).grid(row=0,column=2)
Label(window01,width=20,height=3,textvariable=product,bg='orange',font=('microsoft yahei', 12, 'bold')).grid(row=1,column=3)
Label(window01,width=20,height=3,textvariable=product,bg='orange',font=('microsoft yahei', 12, 'bold')).grid(row=1,column=2)
Entry(window01,width=10,textvariable=user_id,bg='light yellow',font=('microsoft yahei', 16, 'bold')).grid(row=0,column=1)
Label(window01,width=15,text="下次幾次的購買時間:",bg='light yellow',font=('microsoft yahei', 16, 'bold')).grid(row=1,column=0)
Entry(window01,width=10,textvariable=num,bg='light yellow',font=('microsoft yahei', 16, 'bold')).grid(row=1,column=1)
Button(window01,width=20,text="start forecasting",command=get_values,bg='orange',font=('microsoft yahei', 16, 'bold')).grid(row=2,column=0)
Label(window01,width=10,text="下次購買時間:",bg='light yellow',font=('microsoft yahei', 16, 'bold')).grid(row=4,column=1)
Label(window01,width=20,height=15,wraplength =200,textvariable=result01,bg='yellow',font=('microsoft yahei', 16, 'bold')).grid(row=28,column=1)
Label(window01,width=10,text="下次購買地點:",bg='light yellow',font=('microsoft yahei', 16, 'bold')).grid(row=4,column=3)
Label(window01,width=20,height=15,wraplength =200,textvariable=result01,bg='pink',font=('microsoft yahei', 16, 'bold')).grid(row=28,column=3)
Label(window01,width=10,text="第幾次:",bg='light yellow',font=('microsoft yahei', 16, 'bold')).grid(row=4,column=2)
Label(window01,width=20,height=15,wraplength =200,textvariable=result01,bg='sky blue',font=('microsoft yahei', 16, 'bold')).grid(row=28,column=2)
Label(window01,width=20,text="未來是否為流失潛在客戶",bg='light yellow',font=('microsoft yahei', 16, 'bold')).grid(row=24,column=0)
Label(window01,width=5,height=3,wraplength =100,textvariable=result01,bg='red',font=('microsoft yahei', 16, 'bold')).grid(row=28,column=0)
window01.mainloop() 







