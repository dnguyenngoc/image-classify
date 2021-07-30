from fastapi import APIRouter

from api.resources.v1 import image_classify


router = APIRouter()


router.include_router(image_classify.router, prefix="/image-classify",  tags=["Image Classify"])