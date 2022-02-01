from django.urls import path
from . import views

urlpatterns = [
    path('post/getMusic', views.getMusic, name="getMusic"),
    path('', views.home, name="home")
]
