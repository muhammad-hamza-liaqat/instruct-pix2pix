from django.urls import path
from .views import ImageGenView, SDXLImageGenView

urlpatterns = [
    path("generate/", ImageGenView.as_view(), name="image-generate"),
    path("sd-xl-base/", SDXLImageGenView.as_view(), name="sd-xl-generate"),
]
