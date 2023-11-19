import os
import sys
sys.path.append("D:\\private\\project\\pro4_web")
sys.path.append("D:\\private\\project\\pro4_web\\tempProject")
sys.path.append("D:\\private\\project\\pro4_web\\tempProject")

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tempPjt.settings')

from django.contrib import admin

# Register your models here.
from students.models import Student
admin.site.register(Student)

