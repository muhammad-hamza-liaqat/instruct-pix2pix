from django.urls import path
from .views import ImageGenView

urlpatterns = [
    path("generate/", ImageGenView.as_view(), name="image-generate"),
]
