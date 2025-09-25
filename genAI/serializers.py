from rest_framework import serializers

class ImageGenSerializer(serializers.Serializer):
    prompt = serializers.CharField(required=True)
    image = serializers.ImageField(required=True)
