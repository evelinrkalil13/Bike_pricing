from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model_manager import load_model
import pandas as pd
 
app = FastAPI()
templetes = Jinja2Templates(directory="templates")
 
 
@app.get("/")
async def root():
    return templetes.TemplateResponse("index.html", {"request": {},"name":"javier"})
 
@app.post("/calcular_precio", response_class=HTMLResponse)
async def CalcularPrecio(
    request: Request,
    brand: str = Form(...),
    type: str = Form(...),
    frame_material: str = Form(...),
    gear_count: int = Form(...),
    wheel_size: float = Form(...),
    weight_kg: float = Form(...),
):
    
    model= load_model('models/model.pkl')
    df = pd.DataFrame({
    'brand': [brand],
    'type': [type],
    'frame_material': [frame_material],
    'gear_count': [gear_count],
    'wheel_size': [wheel_size],
    'weight_kg': [weight_kg]
})
    print(df)
    print(model)
    prediction = model.predict(df)

    return templetes.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction":prediction[0],
        },
    )
 
 
@app.post("/post")
async def post_metho(data: dict):
    return {"message": "Hola de nuevo", "data": data}
 