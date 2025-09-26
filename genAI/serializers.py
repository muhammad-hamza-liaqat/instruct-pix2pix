from rest_framework import serializers

class ImageGenSerializer(serializers.Serializer):
    prompt = serializers.CharField(required=True)
    image = serializers.ImageField(required=True)

class ImageGenSerializer2(serializers.Serializer):
    prompt = serializers.CharField(
        max_length=500, 
        help_text="The text prompt describing the image to generate"
    )
    image = serializers.ImageField(
        required=False, 
        help_text="Optional initial image for image-to-image generation"
    )