import os
import uuid
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from PIL import Image

from .serializers import ImageGenSerializer
from .pipelines import pipe, upscaler


class ImageGenView(APIView):
    def post(self, request):
        serializer = ImageGenSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        prompt = serializer.validated_data["prompt"]
        image_file = serializer.validated_data["image"]
        upscale = request.query_params.get("hd", "false").lower() == "true"

        #  Balanced step counts
        steps = 35 if upscale else 25

        init_image = Image.open(image_file).convert("RGB")

        try:
            result = pipe(
                prompt=prompt,
                image=init_image.resize((512, 512)),
                num_inference_steps=steps,
                guidance_scale=5.0   #  Subtle edits, keeps face realistic
            )
            generated_image = result.images[0]

            if upscale:
                generated_image = upscaler(
                    prompt=prompt,
                    image=generated_image
                ).images[0]

            media_root = getattr(settings, "MEDIA_ROOT", "media")
            os.makedirs(media_root, exist_ok=True)

            filename = f"{uuid.uuid4().hex}.png"
            save_path = os.path.join(media_root, filename)
            generated_image.save(save_path)

            return Response({
                "message": "Image generated successfully",
                "hd_mode": upscale,
                "steps_used": steps,
                "image_url": request.build_absolute_uri(f"{settings.MEDIA_URL}{filename}")
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
