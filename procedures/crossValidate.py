from sklearn.cross_validation import KFold

class kFold():
	def analyze(self, data, predictors, target, alg, folds=3):
		kf = KFold(data.shape[0], n_folds=folds, random_state=1)
		scores = 0
		for (train, test) in kf:
		    train_predictors = data[predictors].iloc[train,:]
		    train_target = data[target].iloc[train]
		    alg.fit(train_predictors, train_target)
		    test_predictions = alg.predict(data[predictors].iloc[test,:])
		    score = sum([ 1 for result in test_predictions == data[target].iloc[test] if (result)])
		    scores += score

		fullscore = (scores / float(data.shape[0]))
		return fullscore
