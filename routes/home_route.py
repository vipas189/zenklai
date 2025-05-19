from flask import request, render_template, redirect, url_for, flash, current_app
import pandas as pd
from models.training_data import TrainingData
import os
from werkzeug.utils import secure_filename
from extensions import db
import pandas as pd
from services.allowed_file import allowed_file
import shutil
from werkzeug.exceptions import RequestEntityTooLarge
import zipfile
import io


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
        try:
            upload_dir = os.path.join(current_app.root_path, "static/uploads")
            os.makedirs(upload_dir, exist_ok=True)
            zip_file = request.files.get("train_zip")
            if not zip_file or zip_file.filename == "":
                flash("No zip file selected", "error")
                return redirect(url_for("home"))

            if not zip_file.filename.lower().endswith(".zip"):
                flash("Invalid file type. Please upload a .zip file.", "error")
                return redirect(url_for("home"))

            img_folder = os.path.join(upload_dir, "images")
            os.makedirs(img_folder, exist_ok=True)
            csv_processed = 0
            images_processed = 0
            zip_content = io.BytesIO(zip_file.read())

            try:
                with zipfile.ZipFile(zip_content, "r") as zip_ref:
                    zip_filenames = zip_ref.namelist()

                    for filename_in_zip in zip_filenames:
                        sanitized_filename_in_zip = secure_filename(filename_in_zip)

                        if sanitized_filename_in_zip.endswith("/"):
                            continue

                        subfolder_path_in_zip = os.path.dirname(filename_in_zip)
                        base_filename_in_zip = os.path.basename(filename_in_zip)
                        last_subfolder_name = os.path.basename(subfolder_path_in_zip)

                        new_name = (
                            f"{last_subfolder_name}_{base_filename_in_zip}"
                            if last_subfolder_name
                            else base_filename_in_zip
                        )
                        if allowed_file(base_filename_in_zip):
                            target_file_path_base = upload_dir
                        elif allowed_file(base_filename_in_zip, is_image=True):
                            target_file_path_base = img_folder
                        else:
                            print(
                                f"Skipping unsupported file type in zip: {filename_in_zip}"
                            )
                            continue

                        target_file_path = os.path.join(target_file_path_base, new_name)
                        intended_base = os.path.realpath(target_file_path_base)
                        resolved_target = os.path.realpath(target_file_path)

                        if not resolved_target.startswith(intended_base):
                            print(
                                f"Path traversal attempt blocked for: {filename_in_zip}"
                            )
                            flash(
                                f"Blocked potentially unsafe path in zip: {filename_in_zip}",
                                "error",
                            )
                            continue

                        try:
                            with zip_ref.open(filename_in_zip, "r") as internal_file:
                                file_content = internal_file.read()

                            if allowed_file(base_filename_in_zip):
                                temp_csv_path = os.path.join(
                                    upload_dir, f"temp_zip_{new_name}"
                                )
                                with open(temp_csv_path, "wb") as temp_csv_file:
                                    temp_csv_file.write(file_content)

                                try:
                                    for chunk in pd.read_csv(
                                        temp_csv_path, delimiter=";", chunksize=5000
                                    ):
                                        chunk["Filename"] = (
                                            f"{last_subfolder_name}_"
                                            if last_subfolder_name
                                            else ""
                                        ) + chunk["Filename"]

                                        chunk.columns = chunk.columns.str.replace(
                                            ".", "_"
                                        )

                                        existing_filenames = {
                                            row[0]
                                            for row in db.session.execute(
                                                db.select(TrainingData.Filename).where(
                                                    TrainingData.Filename.in_(
                                                        chunk["Filename"].tolist()
                                                    )
                                                )
                                            )
                                        }

                                        chunk = chunk[
                                            ~chunk["Filename"].isin(existing_filenames)
                                        ]

                                        if not chunk.empty:
                                            chunk.to_sql(
                                                "training_data",
                                                con=db.engine,
                                                if_exists="append",
                                                index=False,
                                            )
                                            csv_processed += len(chunk)

                                finally:
                                    if os.path.exists(temp_csv_path):
                                        os.remove(temp_csv_path)

                            elif allowed_file(base_filename_in_zip, is_image=True):
                                with open(target_file_path, "wb") as img_file:
                                    img_file.write(file_content)
                                images_processed += 1

                        except zipfile.BadZipFile:
                            flash("Invalid zip file format.", "error")
                            return redirect(url_for("home"))
                        except Exception as e:
                            flash(
                                f"Error processing file from zip ({filename_in_zip}): {str(e)}",
                                "error",
                            )
                            import traceback

                            print(f"Error processing {filename_in_zip}:")
                            traceback.print_exc()
                            pass

            except zipfile.BadZipFile:
                flash("The uploaded file is not a valid zip file.", "error")
                return redirect(url_for("home"))
            except Exception as e:
                flash(
                    f"An error occurred while processing the zip file: {str(e)}",
                    "error",
                )
                import traceback

                print("Unhandled exception during zip processing:")
                traceback.print_exc()
                return redirect(url_for("home"))

            db.session.commit()

            db_rows = db.session.execute(db.select(TrainingData)).scalars().all()
            expected_image_filenames_in_db = {row.Filename for row in db_rows}
            actual_images_on_disk = set(os.listdir(img_folder))

            removed_db_entries = 0
            for db_row in db_rows:
                if db_row.Filename not in actual_images_on_disk:
                    print(f"Removing DB entry for missing image: {db_row.Filename}")
                    db.session.delete(db_row)
                    removed_db_entries += 1

            removed_images = 0
            for filename_on_disk in actual_images_on_disk:
                if filename_on_disk not in expected_image_filenames_in_db:
                    file_path = os.path.join(img_folder, filename_on_disk)
                    print(f"Removing orphaned image: {file_path}")
                    try:
                        os.remove(file_path)
                        removed_images += 1
                    except PermissionError:
                        print(
                            f"Could not delete orphaned image (permission error): {file_path}"
                        )

            db.session.commit()

            # flash(
            #     f"Processed {csv_processed} CSV records and {images_processed} images from zip. Removed {removed_db_entries} database entries and {removed_images} orphaned images.",
            #     "image_count",
            # )
            return redirect(url_for("home"))

        except RequestEntityTooLarge:
            flash(
                "The uploaded zip file is too large. Please try uploading a smaller zip file or increasing the MAX_CONTENT_LENGTH in your Flask configuration.",
                "error",
            )
            return redirect(url_for("home"))
        except Exception as e:
            flash(
                f"An unhandled error occurred during zip processing: {str(e)}", "error"
            )
            import traceback

            print("Unhandled exception during zip upload:")
            traceback.print_exc()
            return redirect(url_for("home"))

    @app.route("/delete/upload", methods=["POST"])
    def delete_upload():
        db.session.execute(db.delete(TrainingData))
        db.session.commit()
        img_folder = os.path.join(current_app.root_path, "static/uploads/images")
        shutil.rmtree(img_folder)

        return redirect(url_for("home"))
