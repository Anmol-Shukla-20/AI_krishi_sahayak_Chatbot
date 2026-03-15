from flask import Flask, render_template, request, redirect, session, url_for, flash
import joblib
import numpy as np
import os
from dotenv import load_dotenv
from authlib.integrations.flask_client import OAuth
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import requests
from groq import Groq
# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv(override=True)


grok_client = Groq(
    api_key=os.environ.get("GROK_API_KEY")
)


# ------------------------------
# Initialize Flask
# ------------------------------
app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.secret_key = os.environ.get("SECRET_KEY")

# ------------------------------
# Database
# ------------------------------
basedir = os.path.abspath(os.path.dirname(__file__))
# Bind the raw SQLite database directly to the instance folder inside the app's exact directory.
db_path = os.path.join(basedir, 'instance', 'database.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/database.db'
db = SQLAlchemy(app)

# ------------------------------
# User Model
# ------------------------------
class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(200))

with app.app_context():
    db.create_all()

# ------------------------------
# Google OAuth
# ------------------------------
oauth = OAuth(app)

google = oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"}
)

# ------------------------------
# Language Support
# ------------------------------
translations = {

    "en": {
        "title": "Crop Recommendation System",
        "predict": "Predict Crop",
        "upload": "Upload Soil Image",
        "logout": "Logout"
    },

    "hi": {
        "title": "फसल अनुशंसा प्रणाली",
        "predict": "फसल भविष्यवाणी",
        "upload": "मिट्टी की छवि अपलोड करें",
        "logout": "लॉगआउट"
    },

}

# ------------------------------
# Load Crop Prediction Model
# ------------------------------
model_data = joblib.load('crop_model.pkl')

scaler = model_data['scaler']
le = model_data['label_encoder']
model = model_data['model']



# manual routes to be implemented -------------
# Signup
# ------------------------------
# @app.route("/signup", methods=["GET","POST"])
# def signup():

#     if request.method == "POST":

#         email = request.form["email"]
        
#         existing_user = User.query.filter_by(email=email).first()
#         if existing_user:
#             flash('Email already registered. Please login.', 'danger')
#             return redirect("/signup")

#         password = generate_password_hash(request.form["password"])

#         user = User(email=email, password=password)

#         db.session.add(user)
#         db.session.commit()
        
#         flash('Account created successfully! Please login.', 'success')
#         return redirect("/login")

#     return render_template("signup.html")

# ------------------------------
# Login
# ------------------------------
# @app.route("/login", methods=["GET","POST"])
# def login():

#     if request.method == "POST":

#         email = request.form["email"]
#         password = request.form["password"]

#         user = User.query.filter_by(email=email).first()

#         if user and check_password_hash(user.password, password):

#             session["user"] = email

#             return redirect("/dashboard")
#         else:
#             flash('Invalid email or password.', 'danger')

#     return render_template("login.html")






# ------------------------------
#Routes
# ------------------------------

# @app.route("/home")
# def intro():
#     if "user" not in session:
#         return redirect("/login")
#     return render_template("home.html")


@app.route('/')
def home():

    if "user" not in session:
        return redirect("/login")

    lang = session.get("lang", "en")

    return render_template(
        "index_generated.html",
        text=translations[lang]
    )

@app.route("/login")
def login():
    return render_template("login.html")


# Google Login
# ------------------------------
@app.route("/google_login")
def google_login():

    # redirect_uri = url_for("google_authorize", _external=True)
    redirect_uri = "https://ai-krishi-sahayak-chatbot.onrender.com/authorize"
    return google.authorize_redirect(redirect_uri)

@app.route("/authorize")
def google_authorize():

    token = google.authorize_access_token()
    
    # resp = google.get("userinfo")
    resp = google.get("https://www.googleapis.com/oauth2/v2/userinfo")
    user_info = resp.json()
    session["user"] = user_info["email"]
    session["name"] = user_info["name"]

    return redirect("/")
    # user_info = token["userinfo"]
    # session["user"] = user_info["email"]
    # return redirect("/dashboard")

@app.route("/contact")
def contact():
    return render_template("contact.html")


# ------------------------------
# Dashboard
# ------------------------------
@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect("/login")

    return render_template("dashboard.html")

# ------------------------------
# Weather
# ------------------------------
@app.route("/weather", methods=["GET", "POST"])
def weather():

    if "user" not in session:
        return redirect("/login")

    city = "Lucknow"
    
    if request.method == "POST":
        city = request.form.get("city", "Lucknow").strip()

    api_key = os.environ.get("WEATHER_API_KEY")

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    
    # Check if city was found
    if response.status_code != 200:
        flash(f"Could not find weather data for '{city}'. Displaying defaults.", "warning")
        city = "Lucknow"
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        data = requests.get(url).json()
    else:
        data = response.json()

    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    pressure = data["main"]["pressure"]
    wind = data["wind"]["speed"]
    description = data["weather"][0]["description"].title() if data.get("weather") else "Clear"

    return render_template("weather.html",
                           current_city=city,
                           temperature=temperature,
                           humidity=humidity,
                           pressure=pressure,
                           wind=wind,
                           description=description)

# ------------------------------
# Chatbot
# ------------------------------
@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():

    if "user" not in session:
        return redirect("/login")

    # If the frontend is sending an API fetch request (JSON)
    if request.is_json:
        try:
            data = request.get_json()
            user_message = data.get("message", "")

            # Call the Groq API from the Python backend seamlessly
            completion = grok_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an agriculture expert helping farmers with crops, soil, fertilizers and weather advice."
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                temperature=0.5,
                max_completion_tokens=1024,
                top_p=1,
                stream=False
            )
            
            bot_reply = completion.choices[0].message.content
            return {"reply": bot_reply}, 200

        except Exception as e:
            # Send the error text directly back to the frontend for easy reading
            return {"error": str(e)}, 500

    # If the user is just loading the page (GET request)
    return render_template("chatbot.html")
        
# ------------------------------
# Crop Prediction
# ------------------------------
@app.route('/predict', methods=['POST'])
def predict():

    if "user" not in session:
        return redirect("/login")

    try:

        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)

        crop_name = le.inverse_transform(prediction)[0]

        image_file = f"images/{crop_name.lower()}.jpg"

        return render_template(
            'index_generated.html',
            text=translations[session.get("lang", "en")],
            result=crop_name,
            image_file=image_file
        )

    except Exception as e:

        return render_template(
            'index_generated.html',
            text=translations[session.get("lang", "en")],
            result=f"Error: {str(e)}"
        )

# ------------------------------
# Language Change
# ------------------------------
@app.route("/set_language/<lang>")
def set_language(lang):

    session["lang"] = lang

    return redirect("/")

# ------------------------------
# Logout
# ------------------------------
@app.route("/logout")
def logout():

    session.pop("user", None)

    return redirect("/login")


# ------------------------------
# Run App

if __name__ == '__main__':   
    app.run(host='0.0.0.0', port=10000)
