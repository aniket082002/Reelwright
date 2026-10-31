cd backend
if not exist venv python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
start http://localhost:8000
uvicorn main:app
