from fastapi import APIRouter

from api.resources.v1 import image_classify
from api.resources.v1 import es



router = APIRouter()


router.include_router(image_classify.router, prefix="/image-classify",  tags=["Image Classify"])
router.include_router(es.router, prefix="/es",  tags=["Elasticsearch"])
