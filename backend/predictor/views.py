import logging
import traceback
import uuid
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .inference import predict_xray

log = logging.getLogger("predictor")


@csrf_exempt
def predict(request):
    if request.method != "POST":
        return JsonResponse({"error": "Use POST."}, status=405)

    uploaded = request.FILES.get("image")
    if not uploaded:
        return JsonResponse({"error": "No image uploaded. Use form field 'image'."}, status=400)

    upload_dir = Path(settings.MEDIA_ROOT) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # keep only the extension from the client's name (no odd characters / paths)
    ext = Path(uploaded.name).suffix.lower() or ".img"
    name = f"{uuid.uuid4().hex}{ext}"
    path = upload_dir / name

    with open(path, "wb") as destination:
        for chunk in uploaded.chunks():
            destination.write(chunk)

    log.info("UPLOAD original_name=%r bytes=%d saved_as=%s", uploaded.name, uploaded.size, name)

    try:
        result = predict_xray(str(path), str(settings.MEDIA_ROOT))
        result["uploaded_image_url"] = request.build_absolute_uri(
            settings.MEDIA_URL + "uploads/" + name
        )
        return JsonResponse(result)
    except Exception as exc:
        traceback.print_exc()
        return JsonResponse({"error": str(exc)}, status=500)