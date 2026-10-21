from pydantic import BaseModel

class ImageUploadResponse(BaseModel):
    success: bool
    filename: str
    content_type: str
    width: int
    height: int
    message: str | None = None