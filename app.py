import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify
)

from pymongo import MongoClient
from bson.objectid import ObjectId

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.utils import secure_filename

from PIL import Image

import tensorflow as tf

from tensorflow.keras.utils import img_to_array

from tensorflow.keras.applications.efficientnet import preprocess_input

app=Flask(__name__)

app.config["SECRET_KEY"]="supersecretkey"

mongo_uri = os.environ.get(
    "MONGO_URI",
    "mongodb+srv://devilark97_db_user:ZXXuNE1YKwY3DsCA@medai.nhikyij.mongodb.net/medai?appName=medai"
)
client = MongoClient(mongo_uri)
try:
    db = client.get_default_database()
except Exception:
    db = client["medai"]

login_manager=LoginManager()

login_manager.init_app(app)

login_manager.login_view="login"

login_manager.login_message_category="error"

model=joblib.load("disease_model.pkl")

def safe_load_model(path):

    try:

        return tf.keras.models.load_model(
            path,
            compile=False
        )

    except Exception as e:

        print("MODEL LOAD ERROR:",e)

        return None

skin_model=safe_load_model(
    "final_disease_model.keras"
)

class_names=[

    "Acne",

    "Candidiasis",

    "Infestations_Bites",

    "Psoriasis",

    "SkinCancer",

    "Warts"
]

class User(UserMixin):
    def __init__(self, user_id, username, password):
        self.id = user_id
        self.username = username
        self.password = password

    @staticmethod
    def get_by_id(user_id):
        try:
            user_data = db.users.find_one({"_id": ObjectId(user_id)})
            if user_data:
                return User(str(user_data["_id"]), user_data["username"], user_data["password"])
        except Exception as e:
            print("Error loading user by id:", e)
        return None

    @staticmethod
    def get_by_username(username):
        user_data = db.users.find_one({"username": username})
        if user_data:
            return User(str(user_data["_id"]), user_data["username"], user_data["password"])
        return None

@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(user_id)

@app.route("/")
def home():

    slides=[

        {
            "image":"images/l1.png",
            "mobile_image":"images/l1mobile.png"
        },

        {
            "image":"images/l2.png",
            "mobile_image":"images/l2mobile.png"
        },

        {
            "image":"images/l3.png",
            "mobile_image":"images/l3mobile.png"
        }
    ]

    return render_template(
        "index.html",
        user=current_user,
        slides=slides
    )

@app.route("/register",methods=["GET","POST"])
def register():

    if request.method=="POST":

        username=request.form["username"]

        password=request.form["password"]

        existing_user = User.get_by_username(username)

        if existing_user:

            flash(
                "Username already exists!",
                "error"
            )

            return redirect(
                url_for("register")
            )

        hashed_password=generate_password_hash(
            password
        )

        db.users.insert_one({
            "username": username,
            "password": hashed_password
        })

        flash(
            "Registration successful! Please login.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template("register.html")

@app.route("/login",methods=["GET","POST"])
def login():

    if request.method=="POST":

        username=request.form["username"]

        password=request.form["password"]

        user = User.get_by_username(username)

        if user and check_password_hash(
            user.password,
            password
        ):

            login_user(user)

            return redirect(
                url_for("home")
            )

        else:

            flash(
                "Invalid username or password!",
                "error"
            )

    return render_template("login.html")

@app.route("/analyze")
@login_required
def dashboard():

    df=pd.read_csv("dataset.csv")

    symptoms=list(df.columns[:-1])

    return render_template(
        "analyze.html",
        symptoms=symptoms,
        user=current_user
    )

@app.route("/predict",methods=["POST"])
@login_required
def predict():

    try:

        data=request.json

        df=pd.read_csv("dataset.csv")

        df=df.drop(columns=["Unnamed: 133"],errors="ignore")

        symptom_columns=list(
            df.drop("prognosis",axis=1).columns
        )

        input_data=[]

        selected_symptoms=[]

        for symptom in symptom_columns:

            value=int(data.get(symptom,0))

            input_data.append(value)

            if value==1:

                selected_symptoms.append(symptom)

        input_data=np.array([input_data])

        predicted_disease=model.predict(
            input_data
        )[0]

        probability=int(

            np.max(

                model.predict_proba(
                    input_data
                )[0]

            )*100
        )

        db.history.insert_one({
            "disease": predicted_disease,
            "probability": probability,
            "symptom_count": len(selected_symptoms),
            "symptoms": ", ".join(selected_symptoms),
            "prediction_type": "Symptom",
            "image_path": "",
            "date": datetime.utcnow(),
            "user_id": current_user.id
        })

        return jsonify({

            "predicted_disease":predicted_disease,

            "probability":probability
        })

    except Exception as e:

        print("❌ Prediction Error:",e)

        return jsonify({

            "predicted_disease":"Error",

            "probability":0

        }),500

@app.route("/image-disease",methods=["GET","POST"])
@login_required
def image_disease():

    prediction=None

    confidence=None

    image_path=None

    if request.method=="POST":

        if "image" not in request.files:

            flash(
                "No image uploaded!",
                "error"
            )

            return redirect(request.url)

        file=request.files["image"]

        if file.filename=="":

            flash(
                "No image selected!",
                "error"
            )

            return redirect(request.url)

        upload_folder=os.path.join(
            "static",
            "uploads"
        )

        if not os.path.exists(upload_folder):

            os.makedirs(upload_folder)

        filename=secure_filename(
            file.filename
        )

        filepath=os.path.join(
            upload_folder,
            filename
        )

        file.save(filepath)

        try:

            img=Image.open(filepath).convert("RGB")

            img=img.resize((224,224))

            img_array=img_to_array(img)

            img_array=np.expand_dims(
                img_array,
                axis=0
            )

            img_array=preprocess_input(
                img_array
            )

            if skin_model is None:

                flash(
                    "Model failed to load!",
                    "error"
                )

                return redirect(request.url)

            predictions=skin_model.predict(
                img_array,
                verbose=0
            )[0]

            predicted_index=np.argmax(
                predictions
            )

            prediction=class_names[
                predicted_index
            ]

            confidence=round(

                float(
                    predictions[predicted_index]
                )*100,

                2
            )

            image_path=filepath

            db.history.insert_one({
                "disease": prediction,
                "probability": confidence,
                "symptom_count": 0,
                "symptoms": "Image Prediction",
                "image_path": image_path,
                "prediction_type": "Image",
                "date": datetime.utcnow(),
                "user_id": current_user.id
            })

        except Exception as e:

            print("IMAGE PREDICTION ERROR:",e)

            flash(
                "Prediction failed!",
                "error"
            )

            return redirect(request.url)

    return render_template(
        "image_disease.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )

@app.route("/history")
@login_required
def history():

    records = db.history.find({
        "user_id": current_user.id
    }).sort("date", -1)

    history_data = []

    for record in records:
        date_val = record.get("date")
        if isinstance(date_val, datetime):
            date_str = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val) if date_val else ""

        history_data.append({
            "date": date_str,
            "disease": record.get("disease"),
            "probability": record.get("probability"),
            "symptom_count": record.get("symptom_count"),
            "symptoms": record.get("symptoms"),
            "image_path": record.get("image_path"),
            "prediction_type": record.get("prediction_type")
        })

    return render_template(
        "history.html",
        history=history_data
    )

@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect(
        url_for("home")
    )

if __name__=="__main__":

    pass

    app.run(debug=True)