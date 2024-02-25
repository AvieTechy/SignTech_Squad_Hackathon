from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from django.conf.urls import include

from homes.views import (
    index, realtime,fileRequest,
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name="index"),
    path('realtime/', realtime, name="realtime"),
    path('fileRequest/', fileRequest, name="fileRequest"),

]+static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)+static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)


