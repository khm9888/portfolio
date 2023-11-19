import datetime
import bs4
import json
import sys
import pandas as pd

from urllib.request import urlopen
import requests



# 파이썬에서의 날짜 형식으로 바꿔주기
def date_format(d):  
    d = str(d).replace('-', '.')
    yyyy = int(d.split('.')[0])
    mm = int(d.split('.')[1])
    dd = int(d.split('.')[2])
    this_date = datetime.date(yyyy, mm,dd)
    return this_date

#국내 코스피 지수
def domesitc(index_cd,last_page):#bs4 이용해서 가져옴
    
    historical_prices = dict()
    for page_n in range(1,last_page+1):
        naver_index = f"https://finance.naver.com/sise/sise_index_day.nhn?code={index_cd}&page={page_n}"
        # print(naver_index)
        
        source = urlopen(naver_index).read()
        source = bs4.BeautifulSoup(source, "html.parser")
        dates = source.find_all('td', class_='date') # <td class="date"> 태그에서 날짜 수집
        prices = source.find_all('td', class_='number_1') # <td class="number_1> 태그에서 price 수집
        for n in range(len(dates)):

            if dates[n].text.split('.')[0].isdigit():
            # 날짜 처리
                this_date = dates[n].text
                this_date = date_format(this_date)
            # 종가 처리
            this_close = prices[n*4].text
            this_close = this_close.replace(',', '')
            this_close = float(this_close)

        #     # 딕셔너리에 저장
            historical_prices[this_date] = this_close

    return historical_prices

def foreign(symbol,last_page):
    historical_prices=dict()
    for page_n in range(1,last_page+1):
        url=f"https://finance.naver.com/world/worldDayListJson.nhn?symbol={symbol}&fdtc=0"
        # print(url)
        resp = requests.get(url,params={"page":page_n},headers={'User-agent': 'Mozilla/5.0'})
        soup = resp.json()
        # print(soup)
        # print(type(soup))
        for n in range(len(soup)):
            date =pd.to_datetime(soup[n]["xymd"]).date()
            price = float(soup[n]["clos"])
            historical_prices[date] = price


    return historical_prices

def exchange_rate(nation, last_page):
    historical_rates = dict()
    for page_n in range(1,last_page+1):
        exchange_url = f"https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_{nation}&page={page_n}"
        # print(exchange_url)
        
        source = urlopen(exchange_url).read()
        source = bs4.BeautifulSoup(source, "html.parser")
        dates = source.find_all('td', class_='date') # <td class="date"> 태그에서 날짜 수집
        prices = source.find_all('td', class_='num')# <td class="number_1> 태그에서 price 수집
        #print(prices)
        for n in range(len(dates)):

            if dates[n].text.split('.')[0].isdigit():
            # 날짜 처리
                this_date = dates[n].text
                this_date = date_format(this_date)
            # 환율 처리
            this_rate = prices[n*2].text
            this_rate = this_rate.replace(',', '')
            this_rate = float(this_rate)

            # 딕셔너리에 저장
            historical_rates[this_date] = this_rate

    return historical_rates
