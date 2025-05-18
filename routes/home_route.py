from flask import request, render_template, redirect, url_for, flash, current_app
import pandas as pd
from models.training_data import TrainingData
import os
from werkzeug.utils import secure_filename
from extensions import db
import pandas as pd
from services.allowed_file import allowed_file
import shutil


def home_route(app):
    @app.route("/")
    def home():
        db_rows = db.session.execute(db.select(TrainingData)).scalars().all()
        for index, db_row in enumerate(db_rows):
            if index > 2:
                flash(f"+{len(db_rows) - 3} images  ", category="image_count")
                break
            flash(f"{db_row.Filename}, ", category="image_count")

        return render_template("home.html")

    @app.route("/upload/folder", methods=["POST"])
    def upload_folder():
        upload_dir = os.path.join(current_app.root_path, "static/uploads")
        os.makedirs(upload_dir, exist_ok=True)
        files = request.files.getlist("train_files")
        img_folder = os.path.join(upload_dir, "images")
        os.makedirs(img_folder, exist_ok=True)

        for f in files:
            subfolder = os.path.basename(os.path.dirname(f.filename))
            filename = secure_filename(os.path.basename(f.filename))
            new_name = f"{subfolder}_{filename}" if subfolder else filename
            if allowed_file(f.filename):
                df = pd.read_csv(f.stream, delimiter=";")
                df["Filename"] = f"{subfolder}_" + df["Filename"]
                df.columns = df.columns.str.replace(".", "_")
                existing_filenames = {
                    row[0]
                    for row in db.session.execute(
                        db.select(TrainingData.Filename).where(
                            TrainingData.Filename.in_(df["Filename"].tolist())
                        )
                    )
                }

                df = df[~df["Filename"].isin(existing_filenames)]

                if not df.empty:
                    df.to_sql(
                        "training_data", con=db.engine, if_exists="append", index=False
                    )
            elif allowed_file(f.filename, is_image=True):
                f.save(os.path.join(img_folder, new_name))
        db.session.commit()
        db_rows = db.session.execute(db.select(TrainingData)).scalars().all()
        existing_images = set(os.listdir(img_folder))
        for db_row in db_rows:
            if db_row.Filename not in existing_images:
                print(f"Deleted data: {db_row.Filename}")
                db.session.delete(db_row)
            else:
                existing_images.remove(db_row.Filename)
        for filename in existing_images:
            file_path = os.path.join(img_folder, filename)
            try:
                os.remove(file_path)
                print(f"Deleted image: {file_path}")
            except PermissionError:
                pass
                print(f"Could not delete image: {file_path}")
        db.session.commit()
        return redirect(url_for("home"))

    @app.route("/delete/upload", methods=["POST"])
    def delete_upload():
        db.session.execute(db.delete(TrainingData))
        db.session.commit()
        img_folder = os.path.join(current_app.root_path, "static/uploads/images")
        shutil.rmtree(img_folder)

        return redirect(url_for("home"))
