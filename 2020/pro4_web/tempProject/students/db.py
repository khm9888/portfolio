import sys
sys.path.append("D:\\private\\project\\pro4_web\\tempProject")

from students.models import Student
qs = Student(s_name="hong",s_major ="computer",s_age=21,s_grade=2,s_gender="M")
qs.save()