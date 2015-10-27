import procedures.crossValidate as cV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import procedures.neuralNetworks as NN

##Defaults
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"

class Titanic():
	def dataClean(self):
		data = pd.read_csv("./data/kaggle/titanic/train.csv")
		data.loc[data["Sex"] == "male", "Sex"] = 0
		data.loc[data["Sex"] == "female", "Sex"] = 1
		data.loc[pd.isnull(data["Embarked"]), "Embarked"] = 0
		data.loc[data["Embarked"] == "S", "Embarked"] = 0
		data.loc[data["Embarked"] == "C", "Embarked"] = 1
		data.loc[data["Embarked"] == "Q", "Embarked"] = 2
		data.loc[pd.isnull(data["Age"]), "Age"] = data["Age"].median()
		data.loc[pd.isnull(data["Fare"]), "Fare"] = data["Fare"].median()
		return data

	def randomForest(self, predictors=predictors, target=target):
		alg = RandomForestClassifier(random_state=1, n_estimators=20, min_samples_split=2, min_samples_leaf=1)
		cleanData = self.dataClean()
		score = cV.kFold().analyze(cleanData, predictors, target, alg)
		return score

if (__name__ == "__main__"):
	analysis = Titanic()
	rF = analysis.randomForest()
	print("Random Forest Score:\n")
	print rF
	print("Random Forest Score:\n")
	print rF
