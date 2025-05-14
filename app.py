from flask import Flask, redirect, url_for
from config import Config
from extensions import db, migrate
from routes.home_route import home_route
from services.training_data import TrainingData


app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
migrate.init_app(app, db)


@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for("home"))


home_route(app)


if __name__ == "__main__":
    app.run(debug=True)
