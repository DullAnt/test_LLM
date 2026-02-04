from api.app import create_app
import uvicorn

app = create_app()


if __name__ == "__main__":
    print("Starting Uvicorn server...")
    
    
    
    uvicorn.run("api.app:create_app",
    host="127.0.0.1",
    port=8000,
    reload=True,
    factory=True)

