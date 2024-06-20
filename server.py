import logging
import uvicorn
from pydantic import BaseModel

from fastapi import FastAPI

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from phi3.run_phi3 import Phi3


app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

phi3 = Phi3()
phi3.load_model(pretrained_model_path='/workspace/model/phi3', tokenizer_model_path='/workspace/model/phi3')

class Message(BaseModel):
    message: str

@app.get("/test")
async def root():
    return "Hello World"

@app.post("/test-post")
async def post_root(message: Message):
    prompt = message.message
    return "Hello World: " + prompt

@app.post("/inferx")
async def generate_club_response(message: Message):
    """Endpoint for chat requests.
    """

    prompt = message.message
    response_msg = phi3.run_model(prompt)
    print(response_msg)
    
    return response_msg
    
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