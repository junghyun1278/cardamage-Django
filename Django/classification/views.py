from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.views.generic.base import TemplateView
from .inference.segm_inference import segmenty
from .inference.model_inference import classify
from .inference.total_damage import damaged_car
from django.conf import settings
from pathlib import Path
from .forms import UploadImageForm
from django.views import generic
import os
import pandas as pd
import numpy as np

from django.conf import settings as settings



# Create your views here.


def home_view(request):
    return render(request, 'home.html')
    

def classificaion_view(request):
    form = UploadImageForm(request.POST or None, request.FILES)
    result = None
    if form.is_valid():
        image_field = form.cleaned_data['image']
        form.save()
        result = classify(settings.MEDIA_ROOT, image_field.name)
    context = {
        'form':form,
        'result':result
    }
    return render(request, 'classificaion.html', context)


def segmentation_view(request):
    form = UploadImageForm(request.POST or None, request.FILES)
    result = None
    if form.is_valid():
        image_field = form.cleaned_data['image']
        form.save()
        result = segmenty(settings.MEDIA_ROOT, image_field.name)
    context = {
        'form':form,
        'result':result
    }
    return render(request, 'segmentation.html', context)


def damage_view(request):
    form = UploadImageForm(request.POST or None, request.FILES)
    result = None
    if form.is_valid():
        image_field = form.cleaned_data['image']
        form.save()
        result = damaged_car(settings.MEDIA_ROOT, image_field.name)
    context = {
        'form':form,
        'result':result
    }
    return render(request, 'damage.html', context)




from .inference.front import front_car
from .inference.back import back_car
from .inference.left import left_car
from .inference.right import right_car

def front_view(request):
    form = UploadImageForm(request.POST or None, request.FILES)
    result = None
    car_class = None
    if form.is_valid():
        image_field = form.cleaned_data['image']
        form.save()
        result = front_car(settings.MEDIA_ROOT, image_field.name)
        car_class = classify(settings.MEDIA_ROOT, image_field.name)

    context = {
        'form':form,
        'result':result,
        'car_class':car_class
    }
    return render(request, 'last.html', context)

def back_view(request):
    form2 = UploadImageForm(request.POST or None, request.FILES)
    result2 = None
    if form2.is_valid():
        image_field = form2.cleaned_data['image']
        form2.save()
        result2 = back_car(settings.MEDIA_ROOT, image_field.name)

    context = {
        'form2':form2,
        'result2':result2,
    }
    return render(request, 'last.html', context)

def left_view(request):
    form3 = UploadImageForm(request.POST or None, request.FILES)
    result3 = None
    if form3.is_valid():
        image_field = form3.cleaned_data['image']
        form3.save()
        result3 = left_car(settings.MEDIA_ROOT, image_field.name)
    
    context = {
        'form3':form3,
        'result3':result3,
    }
    return render(request, 'last.html', context)

def right_view(request):
    form4 = UploadImageForm(request.POST or None, request.FILES)
    result4 = None
    if form4.is_valid():
        image_field = form4.cleaned_data['image']
        form4.save()
        result4 = right_car(settings.MEDIA_ROOT, image_field.name)

    context = {
        'form4':form4,
        'result4':result4
    }
    return render(request, 'last.html', context)


def total_views(request):
    return render(request, 'damages_view.html')

def prediction_view(request):
    form1 = UploadImageForm(request.POST or None, request.FILES)
    result1 = None
    if form1.is_valid():
        image_field = form1.cleaned_data['image']
        form1.save()
        result1 = damaged_car(settings.MEDIA_ROOT, image_field.name)


    form2 = UploadImageForm(request.POST or None, request.FILES)
    result2 = None
    if form2.is_valid():
        image_field = form2.cleaned_data['image']
        form2.save()
        result2 = damaged_car(settings.MEDIA_ROOT, image_field.name)


    form3 = UploadImageForm(request.POST or None, request.FILES)
    result3 = None
    if form3.is_valid():
        image_field = form3.cleaned_data['image']
        form3.save()
        result3 = damaged_car(settings.MEDIA_ROOT, image_field.name)
    

    form4 = UploadImageForm(request.POST or None, request.FILES)
    result4 = None
    if form4.is_valid():
        image_field = form4.cleaned_data['image']
        form4.save()
        result4 = damaged_car(settings.MEDIA_ROOT, image_field.name)

    context = {
        'form1':form1,
        'result1':result1,
        'form2':form2,
        'result2':result2,
        'form3':form3,
        'result3':result3,
        'form4':form4,
        'result4':result4
    }
    return render(request, 'test_prediction.html', context)


def last_view(request):
    form1 = UploadImageForm(request.POST or None, request.FILES)
    result = None
    car_class = None
    if form1.is_valid():
        image_field = form1.cleaned_data['image']
        form1.save()
        result = front_car(settings.MEDIA_ROOT, image_field.name)
        car_class = classify(settings.MEDIA_ROOT, image_field.name)


    form2 = UploadImageForm(request.POST or None, request.FILES)
    result2 = None
    if form2.is_valid():
        image_field = form2.cleaned_data['image']
        form2.save()
        result2 = back_car(settings.MEDIA_ROOT, image_field.name)


    form3 = UploadImageForm(request.POST or None, request.FILES)
    result3 = None
    if form3.is_valid():
        image_field = form3.cleaned_data['image']
        form3.save()
        result3 = left_car(settings.MEDIA_ROOT, image_field.name)

    form4 = UploadImageForm(request.POST or None, request.FILES)
    result4 = None
    if form4.is_valid():
        image_field = form4.cleaned_data['image']
        form4.save()
        result4 = right_car(settings.MEDIA_ROOT, image_field.name)

    context = {
        'form1':form1,
        'form2':form2,
        'form3':form3,
        'form4':form4,
    }

    return render(request, 'last.html', context)
    
def predict_view(request):
    car_dir = os.path.join(settings.BASE_DIR,'classification/inference')
    car_model = pd.read_csv(car_dir + '/car_model.csv')
    car_model = car_model.loc[0][0]
    context = {
        'car_model':car_model,
    }

    return render(request, 'predict.html', context)