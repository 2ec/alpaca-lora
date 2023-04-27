#import lime
#import sklearn
#import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
#from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
import nltk
nltk.download('punkt')



print("Reading dataset")
df = pd.read_json("results/20000_answered_json_formated.json")

# REMOVE
df = df[:50]
# FINISH


print("Splitting dataset")
# Split dataset in train and val sets, where validation data is set to 30% of the avalible data.
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)

# Reset indices to reflect the new length of the two sets.
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)



def split_df(df, complete_prompt=False):
    """
    Splits off Pandas DataFrame in strings to match input prompt to Alpaca
    """

    df_X_list = []
    df_y_list = []
    df_complete_list = []
    for _, row in df.iterrows():
        instruction_str = str(row['instruction'])
        label_score_list = []
        for label_score in row['input']:
            label, score = label_score
            label = label
            score = round(score, 4)
            label_score_list.append([label, score])
        df_X_list.append(f"instruction: {instruction_str}, input: {label_score_list}")
        df_y_list.append(row["output_answered"])

        if complete_prompt:
            df_complete_list.append(f"instruction: {instruction_str}, input: {label_score_list}, output: {row['output_answered']}")
    
    if complete_prompt:
        return df_X_list, df_y_list, df_complete_list
    return df_X_list, df_y_list

def get_x_and_y_train_test(train_df, val_df, complete_prompt=True):
    X_train, y_train, df_complete_list = split_df(train_df, complete_prompt)
    X_test, y_test = split_df(val_df, complete_prompt=False)

    if complete_prompt:
        return X_train, y_train, X_test, y_test, df_complete_list
    return X_train, y_train, X_test, y_test
    
print("Getting X_train, y_train, X_test, y_test")
X_train, y_train, X_test, y_test, df_complete_list = get_x_and_y_train_test(train_df, val_df, complete_prompt=True)

## vectorize to tf-idf vectors
print("Vectorizing")
# tfidf_vc = TfidfVectorizer(min_df = 10, max_features = 10000, analyzer = "word", ngram_range = (1, 2), stop_words = 'english', lowercase = True)
# train_vc_complete = tfidf_vc.fit_transform(df_complete_list)

# X_train_vc = tfidf_vc.transform(X_train).toarray()
# y_train_vc = tfidf_vc.transform(y_train).toarray()

# X_test_vc = tfidf_vc.transform(X_test).toarray()
# y_test_vc = tfidf_vc.transform(y_test).toarray()

# X_train_vc = nltk.word_tokenize(X_train)
# y_train_vc = nltk.word_tokenize(y_train)

# X_test_vc = nltk.word_tokenize(X_test)
# y_test_vc = nltk.word_tokenize(y_test)

print("Fitting model to data")
# svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train_vc, y_train_vc)
# multi_output_clf = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)


multi_output_clf = MultiOutputRegressor(LinearRegression()).fit(X_train, y_train)




filename = "finalized_model.sav"
print(f"Saving the model as {filename}")
#pickle.dump(multi_output_clf, open(filename, 'wb'))

# performing predictions on the test dataset
print("Predicting data")
accuracy = multi_output_clf.score(X_test, y_test)
print("ACCURACY", accuracy)

y_pred = multi_output_clf.predict(X_test)



print(y_test[0])
print(y_pred[0])

#print("ACCURACY OF THE MODEL: ", metrics.balanced_accuracy_score(y_test, y_pred))
#print(metrics.f1_score(y_true=y_test, y_pred=y_pred))