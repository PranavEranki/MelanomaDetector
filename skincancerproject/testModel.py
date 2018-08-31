from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from moleimages import MoleImages
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from keras.models import model_from_json
import os
import matplotlib.pyplot as plt

def plot_roc(y_test, y_score, title='ROC Curve'):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(title + '.png')
    plt.show()



if __name__ == '__main__':
    try:
        mimg = MoleImages()
        X_test, y_test = mimg.load_test_images(os.path.join(os.getcwd(), 'data_scaled_validation/benign'),
                                                os.path.join(os.getcwd(), 'data_scaled_validation/malign'))
    
        json_file = open(os.path.join(os.getcwd(), "models/FinalModel.json"))
        loaded_model_j = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_j)
        model.load_weights(os.path.join(os.getcwd(), "models/FinalModel.h5"))
        
        print("Model has been loaded from disk")
        
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba >0.5)*1
        print(classification_report(y_test,y_pred))
        plot_roc(y_test, y_pred_proba)
    except:
        print("An error occured")