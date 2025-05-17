from flask import request, render_template, redirect, url_for, current_app
import pandas as pd
from models.training_data import TrainingData
import os
from werkzeug.utils import secure_filename
from extensions import db
import pandas as pd
from services.allowed_file import allowed_file
import shutil
import tempfile
from services.extract_archive import extract_archive


def home_route(app):
    @app.route("/")
    def home():
        return render_template("home.html")

    @app.route("/upload/folder", methods=["POST"])
    def upload_folder():
        upload_dir = os.path.join(current_app.root_path, "static/uploads")
        os.makedirs(upload_dir, exist_ok=True)
        files = request.files.getlist("train_files")
        db_folder = os.path.join(upload_dir, "database")
        img_folder = os.path.join(upload_dir, "images")
        os.makedirs(db_folder, exist_ok=True)
        os.makedirs(img_folder, exist_ok=True)

        for f in files:
            filename = secure_filename(os.path.basename(f.filename))
            ext = filename.lower().rsplit(".", 1)[-1]

            if ext in ("zip", "rar"):
                temp_dir = tempfile.mkdtemp()
                archive_path = os.path.join(temp_dir, filename)
                f.save(archive_path)
                extract_archive(archive_path, temp_dir)

                for ef in os.listdir(temp_dir):
                    ef_path = os.path.join(temp_dir, ef)
                    if os.path.isfile(ef_path):
                        if allowed_file(ef):
                            dest = os.path.join(db_folder, ef)
                            shutil.move(ef_path, dest)
                            df = pd.read_csv(dest, delimiter=";")
                            subfolder = ""  # no subfolder info here, just leave blank or set as needed
                            df["Filename"] = f"{subfolder}_" + df["Filename"]
                            df.to_csv(dest, index=False)
                        elif allowed_file(ef, is_image=True):
                            shutil.move(ef_path, os.path.join(img_folder, ef))
                shutil.rmtree(temp_dir)

            else:
                subfolder = os.path.basename(os.path.dirname(f.filename))
                new_name = f"{subfolder}_{filename}" if subfolder else filename

                if allowed_file(f.filename):
                    path = os.path.join(db_folder, new_name)
                    f.save(path)
                    df = pd.read_csv(path, delimiter=";")
                    df["Filename"] = f"{subfolder}_" + df["Filename"]
                    df.to_csv(path, index=False)
                elif allowed_file(f.filename, is_image=True):
                    f.save(os.path.join(img_folder, new_name))

        return redirect(url_for("home"))

        # if "file" not in request.files:
        #     return redirect(request.url)
        # file = request.files["file"]

        # if file and allowed_file(file.filename):
        #     filename = secure_filename(file.filename)
        #     filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        #     file.save(filepath)

        #     try:
        #         df = pd.read_csv(filepath)
        #         for _, row in df.iterrows():
        #             new_data = TrainingData(
        #                 file_name=row["file_name"],
        #                 width=row["width"],
        #                 height=row["height"],
        #                 roi_x1=row["roi_x1"],
        #                 roi_y1=row["roi_y1"],
        #                 roi_x2=row["roi_x2"],
        #                 roi_y2=row["roi_y2"],
        #                 class_id=row["class_id"],
        #             )
        #             db.session.add(new_data)
        #         db.session.commit()
        #         return "File successfully uploaded and data added to the database."
        #     except Exception as e:
        #         return f"An error occurred: {e}"

        # return "Invalid file format. Please upload a CSV file."

    # @app.route("/upload/image", methods=["POST"])
    # def upload_image():
    #     if "file" not in request.files:
    #         return redirect(request.url)
    #     file = request.files["file"]

    #     if file and allowed_file(file.filename):
    #         filename = secure_filename(file.filename)
    #         filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    #         file.save(filepath)

    #         try:
    #             df = pd.read_csv(filepath)
    #             for _, row in df.iterrows():
    #                 new_data = TrainingData(
    #                     file_name=row["file_name"],
    #                     width=row["width"],
    #                     height=row["height"],
    #                     roi_x1=row["roi_x1"],
    #                     roi_y1=row["roi_y1"],
    #                     roi_x2=row["roi_x2"],
    #                     roi_y2=row["roi_y2"],
    #                     class_id=row["class_id"],
    #                 )
    #                 db.session.add(new_data)
    #             db.session.commit()
    #             return "File successfully uploaded and data added to the database."
    #         except Exception as e:
    #             return f"An error occurred: {e}"

    #     return "Invalid file format. Please upload a CSV file."
