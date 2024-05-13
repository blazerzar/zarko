from fastapi import FastAPI, UploadFile, Form, File
from random import randint

app = FastAPI()

#allow cors
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_photo")
async def process_photo(photo: UploadFile = File(...), temperature: float = Form(...)):
    # Process the photo and temperature here
    # You can save the photo, perform some image processing, etc.
    
    # Save the photo
    photo_name = photo.filename
    with open(photo_name, "wb") as buffer:
        buffer.write(photo.file.read())    
    print("Photo saved as: ", photo_name)
    print("Temperature: ", temperature)

    # Generate a random number
    random_number = randint(1, 100)

    return {"result": random_number}

#to run this use uvicorn main:app --reload --port 8000
