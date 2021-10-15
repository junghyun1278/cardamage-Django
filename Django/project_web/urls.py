"""project_web URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import include, url
from classification.views import classificaion_view, home_view, segmentation_view, damage_view, prediction_view
from classification.views import total_views ,front_view, back_view, right_view, left_view
from classification.views import last_view, predict_view
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', home_view, name='home'),
    path('damage/', damage_view, name='damage'),
    path('classification/', classificaion_view, name='classification'),
    path('segmentation/', segmentation_view, name='segmentation'),

    path('total/', total_views, name='total'),
    path('front/', front_view, name='front'),
    path('back/', back_view, name='back'),
    path('left/', left_view, name='left'),
    path('right/', right_view, name='right'),


    path('last/', last_view, name='last'),
    path('predict/', predict_view, name='predict'),
    
    
    path('prediction/', prediction_view, name='prediction'),



    path('admin/', admin.site.urls),
]