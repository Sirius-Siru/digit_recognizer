from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def evaluate(y_test, y_pred):
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy = {accuracy*100:.2f}')
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
def showWrongCase(model, X_test, y_test, proba = None):
    
    probs = model.predict_proba(X_test) if proba is None else proba
    y_pred = np.argmax(probs, axis=1) 
    confidence = np.max(probs, axis=1)
    wrong_idx = np.where(y_pred != y_test)[0]
    wrong_conf = confidence[wrong_idx]
    top_wrong_idx = wrong_idx[np.argsort(-wrong_conf)]
    top20 = top_wrong_idx[:20]
    
    for i, idx in enumerate(top20):
        plt.subplot(4, 5, i+1)
        
        img = X_test[idx][:784].reshape(28, 28) 
        plt.imshow(img, cmap='gray')
        
        plt.title(f"Idx: {idx}\nTrue: {y_test.iloc[idx]}\nPred: {y_pred[idx]}\nConf: {confidence[idx]:.2f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()