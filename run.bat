call "C:\Program Files (x86)\Intel\openvino_2022\setupvars.bat"
set FLASK_APP = app
set FLASK_ENV = development
python -m flask run --reload