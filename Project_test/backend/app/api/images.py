import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
from app.schemas.image import ImageUploadResponse

router = APIRouter(prefix="/images", tags=["images"])

# Create a temporary uploads directory
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"image/jpeg", "image/png", "image/webp"}

@router.post("/upload", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    # 1. Validate file type
    if file.content_type not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        # 2. Read file content
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # 3. Get metadata
        width, height = img.size
        
        # 4. Generate safe filename
        extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        safe_filename = f"{uuid.uuid4()}.{extension}"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)

        # 5. Save temporarily
        with open(file_path, "wb") as f:
            f.write(contents)

        return ImageUploadResponse(
            success=True,
            filename=safe_filename,
            content_type=file.content_type,
            width=width,
            height=height,
            message="Image uploaded successfully"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
```

Now, I'll register this router in the main FastAPI app.

```tool
TOOL_NAME: edit_existing_file
BEGIN_ARG: filepath
backend/app/main.py