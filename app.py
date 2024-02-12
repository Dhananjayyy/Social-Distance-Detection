import functools
import time
import cv2
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, app,render_template, g, session, flash
from flask.globals import request
from flask.helpers import url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug import Response
from flask_wtf import FlaskForm
from wtforms.fields import (StringField,PasswordField,SubmitField)
from wtforms.validators import DataRequired
from werkzeug.utils import redirect
import json
from common import prepare_image
from common import draw_bounding_boxes
from openvino.inference_engine import IECore

# Initialize Flask 
app = Flask(__name__)


# Initialize database
app.config['SECRET_KEY'] = "asdfhalsdlkjflkjaklklajsofennfkjnsjkndkjfnkjndsjf"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
db = SQLAlchemy(app)

class UserDBModel(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    username = db.Column(db.String(500))
    password = db.Column(db.String(500))
    timestamp = db.Column(db.DateTime,default=datetime.now)


db.create_all()
db.session.commit()


@app.before_request
def load_user():
    userid = session.get('userid')
    user = UserDBModel.query.filter_by(id= userid).first()
    g.user = user


def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            flash("Unauthorized", "danger")
            return redirect("/login")
        return view(**kwargs)

    return wrapped_view


# Load model (FP32 is available)
model_name = "ssd_mobilenet_v2_coco"
model_precision = "FP16"
model_xml_path = f"models/{model_name}/{model_precision}/{model_name}.xml"
model_bin_path = f"models/{model_name}/{model_precision}/{model_name}.bin"

# Initialize inference engine
ie = IECore()

# Read inference engine network
network = ie.read_network(model_xml_path, model_bin_path)

# Find input shape, layout and size
input_name = next(iter(network.input_info))
input_data = network.input_info[input_name].input_data
input_shape = input_data.shape # [1, 3, 300, 300]
input_layout = input_data.layout # NCHW
input_size = (input_shape[2], input_shape[3]) # (300, 300)

# Load network (Change CPU to GPU to utilize Intel Integrated GPU)
device = "CPU" # CPU
exec_network = ie.load_network(network=network, device_name=device, num_requests=1)

# Load classes
classes_path = f"models/{model_name}/classes.json"
with open(classes_path) as f:
    classes = f.read()
classes = json.loads(classes) 



def get_frames():
    # Initialize camera
    cap = cv2.VideoCapture(0)

    # Initialize FPS variables
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        # Capture the frames from the webcam
        ret, frame = cap.read()
        # Get FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))

        # We are using the prepare_image function to preprocess the frames.
        frame_prepared = prepare_image(frame, target_size=input_size, target_layout=input_layout)

        # We are using the inference engine to get the detections
        output = exec_network.infer({input_name: frame_prepared})
        detections = output["DetectionOutput"]

        # We are using the draw_bounding_boxes function to draw the bounding boxes on the frames.     
        image = draw_bounding_boxes(frame, detections, classes)
        cv2.putText(image, fps, (7, 70), cv2.FONT_HERSHEY_PLAIN, 3, (100, 255, 0), 3, cv2.LINE_AA)     
        cv2.imshow("Social Distance Detector", image)
        
        # Using the cv2.imencode function to encode the frames in JPEG format.
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        # Using the yield keyword to return the frames to the main function.
        yield (b'--frame\r\n'
                b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n')


# This route renders the social-distance-servilance-live.html template.
@app.route("/camera/social/distance/detection",methods=["GET"])
def camera_social_social_detection():
    return render_template('social-distance-servilance-live.html')

# We are creating a route called /stream/video
# The get_frames() function is called every time the browser requests the /stream/video route.
# The get_frames() function  is used to send the video frames to the browser.
@app.route("/stream/video",methods=["GET"])
def gvview():
    return Response(get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# We're importing the FlaskForm class from the flask_wtf module.
# We're creating a class called LoginUserForm and RegisterUserForm that inherits from FlaskForm.
# We're creating a username and password field that takes a string.
class LoginUserForm(FlaskForm):
    username = StringField("Username",validators=[DataRequired()])
    password = PasswordField("Password",validators=[DataRequired()])
    submit = SubmitField("Login")

class RegisterUserForm(FlaskForm):
    username = StringField("Username",validators=[DataRequired()])
    password = PasswordField("Password",validators=[DataRequired()])
    submit = SubmitField("Register")


# Render home.html for the root '/' route
@app.route("/",methods=['GET','POST'])
def home():
    return render_template("home.html")

# Render dashboard.html for the '/dashboard'  route
@app.route("/dashboard",methods=['GET','POST'])
@login_required
def dashboard():
    return render_template("dashboard.html")

# Render login.html for the '/login' route
@app.route("/login",methods=['GET','POST'])
def login():
    if g.user:
        return redirect(url_for('dashboard'))
    form = LoginUserForm()
    if form.validate_on_submit():
        user = UserDBModel.query.filter_by(username= form.username.data).first()
        # print(user.username)
        
        if user and check_password_hash(user.password, form.password.data): 
            session.clear()
            session['userid'] = user.id
            flash('Login Successful', "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Please check username and password",'danger')
            return redirect(url_for('login'))
    return render_template("login.html",form=form)

# Render register.html for the '/register' route
@app.route("/register",methods=['GET','POST'])


# Register function
def register():
    # If the user is already logged in, redirect them to the dashboard.
    if g.user:
        return redirect(url_for('dashboard'))

    # Create an instance of the RegisterUserForm class.
    form = RegisterUserForm()  

    # If the form is submitted and validates, check if the username is already taken.
    if form.validate_on_submit():
        user = UserDBModel.query.filter_by(username= form.username.data).first()

        # If the username is not taken, create a new user and add it to the database and redirect to 'login.html' page.
        if user == None:
            user = UserDBModel(
                username=form.username.data,
                password=generate_password_hash(form.password.data)
            )
            db.session.add(user)
            db.session.commit()
            flash("User created Successfuly!", "success")
            return redirect(url_for('login'))

        # If the username is taken, flash a message and redirect to the register page.
        else:
            flash("Username not available!", "danger")

        return redirect(url_for('register'))
    # If the form is not submitted or does not validate, render the register.html template.
    return render_template("register.html",form=form)

# Render logout.html for the '/logout' route
@app.route("/logout")
def logout():
    # Clear Flask session
    session.clear()
    # Flash logout message and return to homepage
    flash(f"Logout Success!", 'success')
    return redirect(url_for("home"))
