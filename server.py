import logging
import uvicorn
from pydantic import BaseModel

from fastapi import FastAPI

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from run_phi3 import CustomPipeline


app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=settings.BACKEND_CORS_ORIGINS,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

logger = logging.getLogger(__name__)
# .env variables can be validated and accessed from the config, here to set a log level
logging.basicConfig(level='INFO')

customPipeline = CustomPipeline()
# customPipeline.load_model(pretrained_model_path='/workspace/model/phi3', tokenizer_model_path='/workspace/model/phi3')
customPipeline.load_flash_model(pretrained_model_path='/workspace/model/phi3', tokenizer_model_path='/workspace/model/phi3')

class Message(BaseModel):
    message: str

@app.get("/test")
async def root():
    return "Hello World"

@app.post("/test-post")
async def post_root(message: Message):
    prompt = message.message
    return "Hello World: " + prompt

@app.post("/inferx", response_class=StreamingResponse)
async def generate_club_response(message: Message) -> StreamingResponse:
    """Endpoint for chat requests.
    It uses the StreamingConversationChain instance to generate responses,
    and then sends these responses as a streaming response.
    :param data: The request data.
    """

    prompt = message.message
    response_msg = customPipeline.run_model(prompt)
    print(response_msg)
    
    return StreamingResponse(
        response_msg,
        media_type="text/event-stream",
    )

@app.post("/inferx-flash", response_class=StreamingResponse)
async def generate_club_response(message: Message) -> StreamingResponse:
    """Endpoint for chat requests.
    It uses the StreamingConversationChain instance to generate responses,
    and then sends these responses as a streaming response.
    :param data: The request data.
    """

    prompt = message.message
    response_msg = customPipeline.run_flash_model(prompt)
    print(response_msg)
    
    return StreamingResponse(
        response_msg,
        media_type="text/event-stream",
    )
    
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://0.0.0.0:8000/epoch-ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)