import pandas as pd

def submit(X, y, test, model):
    model.fit(X, y)
    submission = model.predict(test)
    submission = pd.DataFrame(submission, columns=['Label'])
    submission.insert(0, 'ImageId', range(1, len(submission)+1))
    submission.to_csv('submission.csv', index = False)