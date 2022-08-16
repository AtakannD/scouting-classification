import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Task 1

df_att = pd.read_csv(r"C:\Users\atakan.dalkiran\PycharmProjects\Scouting with Machine Learning\scoutium_attributes.csv",
                     sep=";")
df_potlab = pd.read_csv(
    r"C:\Users\atakan.dalkiran\PycharmProjects\Scouting with Machine Learning\scoutium_potential_labels.csv", sep=";")

df = pd.merge(df_att, df_potlab, how='left', on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])
df = df[df["position_id"] != 1]
df = df[df["potential_label"] != "below_average"]
pt_df = pd.pivot_table(df, values="attribute_value", columns="attribute_id",
                       index=["player_id", "position_id", "potential_label"])
pt_df = pt_df.reset_index(drop=False)
pt_df.info()
pt_columns = pt_df.columns.map(str)

# Task 2

le = LabelEncoder()
pt_df["potential_label"] = le.fit_transform(pt_df["potential_label"])
num_cols = pt_df.columns[3:]
scaler = StandardScaler()
pt_df[num_cols] = scaler.fit_transform(pt_df[num_cols])

# Task 3

y = pt_df["potential_label"]
X = pt_df.drop(["potential_label", "player_id"], axis=1)

models = [('LR', LogisticRegression()),
          ("KNN", KNeighborsClassifier()),
          ("CART", DecisionTreeClassifier()),
          ("RF", RandomForestClassifier()),
          ("Adaboost", AdaBoostClassifier()),
          ("GBM", GradientBoostingClassifier()),
          ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
          ("LightGBM", LGBMClassifier()),
          ("CatBoost", CatBoostClassifier(verbose=False))]

for name, model in models:
    print(name)
    for score in ["accuracy","roc_auc", "f1", "precision", "recall"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
        print(score+" score:"+str(cvs))


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value", ascending=False))
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


model = LGBMClassifier()
model.fit(X, y)
plot_importance(model, X)
