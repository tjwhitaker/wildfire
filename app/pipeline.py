from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from sklearn.preprocessing import MultiLabelBinarizer

class AttributeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class CustomBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, class_labels):
        self.class_labels = class_labels
    def fit(self, X, y=None,**fit_params):
        return self
    def transform(self, X):
        return MultiLabelBinarizer(classes=self.class_labels).fit_transform(X)

class FullPipeline():
    def __init__(self):
        self.pipeline = joblib.load('models/pipeline.pkl')

    def prepare_data(self, data):
        return self.pipeline.transform(data)
