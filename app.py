from flask import Flask, redirect, url_for
from config import Config
from extensions import db, migrate
from routes.home_route import home_route
from models.training_data import TrainingData


app = Flask(__name__)
app.config.from_object(Config)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
db.init_app(app)
migrate.init_app(app, db)


@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for("home"))


home_route(app)
print(app.config["MAX_CONTENT_LENGTH"])


if __name__ == "__main__":
    app.run(debug=True)
