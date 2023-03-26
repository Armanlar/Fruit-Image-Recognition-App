from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('',views.index,name='index'),
    path('api/upload', views.upload_file, name='upload'),
    path('api/csrf_token/', views.get_csrf_token, name='get_csrf_token'),
    path('predict/', views.predict_fruit, name='predict_fruit')]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
