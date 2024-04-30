from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("videoAssessment", views.videoAssessment, name="videoAssessment"),
    path("featureAssessment", views.featureAssessment, name="featureAssessment"),
]
