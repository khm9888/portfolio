from bs4 import BeautifulSoup
import requests
import re
import sys
import pprint


List=[]
# 네이버 뉴스 url을 입력합니다.
url="https://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=100&oid=015&aid=0004377336"

oid=url.split("oid=")[1].split("&")[0]#001
# print("oid")
# print(oid)
aid=url.split("aid=")[1]#0010079145&ptype=052
# print("aid")
# print(aid)
page=1    
header = {
    "User-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36",
    "referer":url,
    
} 
while True :
    c_url="https://apis.naver.com/commentBox/cbox/web_neo_list_jsonp.json?ticket=news&templateId=default_society&pool=cbox5&_callback=jQuery1707138182064460843_1523512042464&lang=ko&country=&objectId=news"+oid+"%2C"+aid+"&categoryId=&pageSize=20&indexSize=10&groupId=&listType=OBJECT&pageType=more&page="+str(page)+"&refresh=false&sort=FAVORITE" 
# 파싱하는 단계입니다.
    r=requests.get(c_url,headers=header)
    cont=BeautifulSoup(r.content,"html.parser")    
    total_comm=str(cont).split('comment":')[1].split(",")[0]
   
    match=re.findall('"contents":([^\*]*),"userIdNo"', str(cont))
# 댓글을 리스트에 중첩합니다.
    List.append(match)
# 한번에 댓글이 20개씩 보이기 때문에 한 페이지씩 몽땅 댓글을 긁어 옵니다.
    if int(total_comm) <= ((page) * 20):
        break
    else : 
        page+=1


# 여러 리스트들을 하나로 묶어 주는 함수입니다.
def flatten(l):
    flatList = []
    for elem in l:
        # if an element of a list is a list
        # iterate over this list and add elements to flatList 
        if type(elem) == list:
            for e in elem:
                flatList.append(e)
        else:
            flatList.append(elem)
    return flatList


# 리스트 결과입니다.
comments = flatten(List)

print("len(comments)")
print(len(comments))

name=__file__.split("\\")[-1]
path=__file__[:-len(name)]
#추가할때는 a
with open(f"{path}/words.txt","a",encoding="utf-8") as f:
    for i in comments:
        f.write(f"{i}\n")

from collections import Counter 
from konlpy.tag import Okt  
okt=Okt()
words=[]
for i in comments:
    words+=okt.nouns(i)

print("len(words)")
print(len(words))
count=Counter(words)
print(count.most_common(100))#가장 많이 나오는 얘들
texts = count.elements()

tx=" ".join(texts)


# tx = "파이썬 워드클라우드 파이썬 좋아 워드클라우드 파이썬 라이브러리 좋아 파이썬 워드클라우드 예시 워드클라우드 우한 폐렴 조심 데이터 분석 우한 워드클라우드 중국 박쥐 감염 코로나바이러스"

library(devtools)
devtools::install_github("lchiffon/wordcloud2")

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
 
stopwords = set(STOPWORDS) 
stopwords.add('일본') 
stopwords.add('길') 
stopwords.add('기업')
stopwords.add('그냥')
stopwords.add('나라')

wordcloud = WordCloud(font_path='C:\Windows\Fonts/gulim.ttc', background_color='white',stopwords=stopwords).generate(tx)
plt.figure(figsize=(10,10)) #이미지 사이즈 지정
plt.imshow(wordcloud, interpolation='lanczos') #이미지의 부드럽기 정도
plt.axis('off') #x y 축 숫자 제거
plt.show() 
plt.savefig()

""" 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt 


wordcloud = WordCloud(font_path='font/NanumGothic.ttf',stopwords=stopwords,background_color='white').generate(text)


# print(set(STOPWORDS))
# stopwords = set(STOPWORDS) 
# stopwords.add('워드클라우드') 
 
 """
 