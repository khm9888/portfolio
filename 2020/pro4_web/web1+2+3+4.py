import markdown

#http를 이용해서 통신

#client-> request -> server
#client <- response <- server

#client - web, app


''' 
cookie - 웹 서버가 브라우저에게 지시하여 사용자의 
로컬 컴퓨터에 파일 또는 메모리에 저장하는 
작은 기록 정보 파일

session -  HTTP Session id를 식별자로 구별하여 
데이터를 사용자의 브라우저에 쿠키형태가 아닌 접속한 
서버 DB에 정보를 저장 합니다.

response -웹브라우저(클라이언트)의 요청에 응답하는것

request - 웹브라우저(클라이언트)를 통해 
서버에 어떤 정보를 요청하는 것
 '''

#cookie - 과거기록을 통해서
#session

#get - query-string-보안취약
#post - 요청메세지(백그라운드, header , body )

#url
#rest url <-장고 

# 쿼리가 바꿈에 따라 만들어진 함수를 써서 맵핑 - url mapping

#client- -> server(웹서버(정적) -> 앱 서버(동적)) -> DB

##################################################################

#2장

#django 설치

#장고를 원하는 위치로 가서 시작한다.
 
#cmd 명령어 필요 -> 상위 이동 cd ../

#cd를 통해서 간다.

# 프로젝트 폴더 생성
#django-admin startproject tempPjt

# 해당 프로젝트 폴더 들어가서 app 생성 -> cd tempPjt
#python manage.py startapp students

#폴더 이름 변경(이름 헷갈리기 때문에)
#move tempPjt tempProject

#mvc 패턴(서버) - spring*
#client ->(request) control -> model ->db  
#client -> control -> view(클라이언트한테 보여줘야하는 화면)

#mvt 패턴 - django
#client ->(request) view -> model ->db  
#client -> view -> templete(클라이언트한테 보여줘야하는 화면)

#templte은 따로 만들어서 호출하는편이다.

###############################################################

#3장

#client->URLConf(urls.py)->view

#urls.py

#views

#def function(request):
#   return HttpResponse()

#models.py

#ORM 방식

#templete - settings.py
#templetes 폴더를 생성

#settings.py -> installed_apps 에 추가하는 앱의 apps.py의 class 명을 추가해야한다.

#debug = true # 개발중일 땐 true, 상용화 하기 전 // false - 상용화 모드

#allowed_hosts 에 고유ip주소를 넣어야함. debug, true라면 개인 사설 ip를 넣어 테스트 할 수 있다.

#장고의 장점 - 다른 프레임워크와 달리
#1) 관리자모드 자동생성 - 그룹테이블을 작성하긴 해야함 ->

#-> 사용자 및 그룹 테이블 생성
#python manage.py migrate

#-> 데이터베이스 변경사항 반영
#python manage.py makemigrations
#python manage.py migrate

#관리자 계정 생성 및 서버 구동
#python manage.py createsuperuser
#이름 : 
#메일주소
#password

#파이썬은 간단한 서버를 제공해준다.
#python manage.py runserver
#python manage.py runserver 0.0.0.0:8000

#브라우저 127.0.0.1:8000
#브라우저 127.0.0.1:8000/admin
#현재는 user // user

############################################################################

#4장.데이터베이스 ORM

#프로젝트 생성-> 1~3까지 한 내용-? 테이블 생성

#manage.py를 이용해서 서버 접속