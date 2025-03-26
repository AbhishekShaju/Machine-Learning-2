from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('polynomial/', views.polynomial, name='polynomial'),
    path('logistic/', views.logistic, name='logistic'),
    path('knn/', views.knn, name='knn'),
    path('predict/polynomial/', views.predict_polynomial, name='predict_polynomial'),
    path('predict/logistic/', views.predict_logistic, name='predict_logistic'),
    path('predict/knn/', views.predict_knn, name='predict_knn'),
] 