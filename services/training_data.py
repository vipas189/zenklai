from extensions import db


class TrainingData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(250), nullable=False)
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    roi_x1 = db.Column(db.Integer, nullable=False)
    roi_y1 = db.Column(db.Integer, nullable=False)
    roi_x2 = db.Column(db.Integer, nullable=False)
    roi_y2 = db.Column(db.Integer, nullable=False)
    class_id = db.Column(db.Integer, nullable=False)
