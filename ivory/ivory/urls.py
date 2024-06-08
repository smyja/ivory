from django.urls import path
from ivory import views

urlpatterns = [path("", views.index, name="index")]
