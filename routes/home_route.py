from flask import request, render_template, redirect, url_for
from services.training_data import TrainingData
import os
from werkzeug.utils import secure_filename
from extensions import db
import pandas as pd
from config import Config


def home_route(app):
    @app.route("/")
    def home():
        return render_template("home.html")

    # Function to check allowed file extensions
    def allowed_file(filename):
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in Config.DATABASE_ALLOWED_EXTENSIONS
        )

    # Route to display the upload form
    @app.route("/upload_form")
    def upload_form():
        return render_template("upload_form.html")

    # Route to handle the uploaded file
    @app.route("/upload", methods=["POST"])
    def upload_file():
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]

        # Check if the file is valid
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Process CSV file and insert data into the database
            try:
                df = pd.read_csv(filepath)
                for _, row in df.iterrows():
                    # Assuming your columns match the ones in the CSV
                    new_data = TrainingData(
                        file_name=row["file_name"],
                        width=row["width"],
                        height=row["height"],
                        roi_x1=row["roi_x1"],
                        roi_y1=row["roi_y1"],
                        roi_x2=row["roi_x2"],
                        roi_y2=row["roi_y2"],
                        class_id=row["class_id"],
                    )
                    db.session.add(new_data)
                db.session.commit()
                return "File successfully uploaded and data added to the database."
            except Exception as e:
                return f"An error occurred: {e}"

        return "Invalid file format. Please upload a CSV file."
