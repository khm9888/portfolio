from urllib.request import urlopen
from pandas import DataFrame 
from bs4 import BeautifulSoup

myencoding = 'utf-8'
myparser = 'html.parser'
url = 'https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=cur'
html = urlopen( url )
soup = BeautifulSoup(html, 'html.parser')


totallist = []
mytrs = soup.find_all('tr')
no = 0

repeat = 2

for one_tr in mytrs :
#     print(one_tr)
    mytd = one_tr.find('td', attrs={'class':'title'})
    if( mytd != None ) :
        no += 1
        newno = '0' * repeat + str(no)
        newno = newno[len(newno) - repeat : len(newno)]

        mytag = mytd.find('div', attrs={'class':'tit5'})
        title = mytag.a.string    
        point = one_tr.find('td', attrs={'class':'point'}).string        
        
        # select 메소드는 요소가 1개인 리스트로 반환해준다.
        updown = one_tr.select_one('img:nth-of-type(3)')
        #print(type(updown))
        if updown['alt'] == 'down' :
            up_down = '하강'
        elif updown['alt'] == 'up' :
            up_down = '상승'
        else :  
            up_down = '--'
        change = one_tr.find('td', attrs={'class':'range ac'})
        if ( change != None ) : # None이면 '신규' 영화이다.
            change = change.string 
        else :
            change = '신규'
             
        subtuple = tuple([newno, title, point, up_down, change])
        totallist.append(subtuple)

print( totallist )

mycolumn = ['순위', '영화 제목', '평점', '변동', '변동값']

myframe = DataFrame( totallist, columns = mycolumn )
filename = 'bsResult03.csv'
myframe.to_csv( filename, index=False)

print('\n# finished')