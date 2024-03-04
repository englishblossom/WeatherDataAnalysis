
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from prettytable import PrettyTable
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
from sklearn import tree
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC


# %%
def Classification():
    # cleaning dataset
    df = pd.read_csv('weatherAUS.csv')
    print((df.isna().sum()) / len(df) * 100)
    sunshine_median = df['Sunshine'].median()
    evaporation_median = df['Evaporation'].median()
    cloud9am_mode = df['Cloud9am'].mode()
    cloud3pm_mode = df['Cloud3pm'].mode()
    df['Sunshine'] = df['Sunshine'].fillna(sunshine_median)
    df['Evaporation'] = df['Evaporation'].fillna(evaporation_median)
    df['Cloud9am'] = df['Cloud9am'].fillna(cloud9am_mode[0])
    df['Cloud3pm'] = df['Cloud3pm'].fillna(cloud3pm_mode[0])
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = numerical_cols.drop(['Rainfall', 'Cloud9am', 'Cloud3pm'])

    for col in numerical_cols:
        median = df[col].median()
        df[col].fillna(median, inplace=True)
    mode_temp = df['WindGustDir'].mode()
    df['WindGustDir'] = df['WindGustDir'].fillna(mode_temp[0])
    mode_temp = df['WindDir9am'].mode()
    df['WindDir9am'] = df['WindDir9am'].fillna(mode_temp[0])
    mode_temp = df['WindDir3pm'].mode()
    df['WindDir3pm'] = df['WindDir3pm'].fillna(mode_temp[0])
    df.dropna(inplace=True)
    print("after cleaning rows that have null values:\n", (df.isnull().sum()) / len(df) * 100)
    pd.set_option('display.max_columns', None)
    print("Cleaned Dataset: \n", df.head(5))
    cleaned_df = df.copy()
    # feature engineering
    df['TempRange'] = df['MaxTemp'] - df['MinTemp']
    df['HumidityTemp9am'] = df['Humidity9am'] * df['Temp9am']
    df['HumidityTemp3pm'] = df['Humidity3pm'] * df['Temp3pm']
    df['PressureChange'] = df['Pressure3pm'] - df['Pressure9am']
    df['CloudCoverChange'] = df['Cloud3pm'] - df['Cloud9am']
    df['WindSpeedChange'] = df['WindSpeed3pm'] - df['WindSpeed9am']
    df.drop(inplace=True, columns=['Date'])

    all_probabilities = []
    model_labels = []

    all_model_metrics = []

    # One-hot encoding
    # %%
    encoding_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
    df_resampled1 = df.drop(columns=encoding_columns)
    df_resampled1.drop(columns=['RainTomorrow', 'RainToday'], inplace=True)
    df_encoded = df[['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']]
    df_encoded = pd.get_dummies(df_encoded, columns=encoding_columns, drop_first=True)
    df_encoded['RainTomorrow'] = df_encoded['RainTomorrow'].map({'No': 0, 'Yes': 1})
    df_encoded['RainToday'] = df_encoded['RainToday'].map({'No': 0, 'Yes': 1})
    df_encoded = df_encoded.astype(int)
    excluded_columns = ['RainTomorrow']
    encoded_columns = [col for col in df_encoded.columns if col not in excluded_columns]
    df_resampled1 = pd.concat([df_resampled1, df_encoded], axis=1)
    Y = df_resampled1['RainTomorrow']
    df_resampled1.drop(columns=['RainTomorrow'], inplace=True)

    # Normalization
    X_train, X_test, Y_train, Y_test = train_test_split(df_resampled1, Y,
                                                        shuffle=True, test_size=0.2, stratify=Y, random_state=5805)
    X_train_standardized = X_train.copy()
    X_test_standardized = X_test.copy()

    min_max_scaler = MinMaxScaler()

    X_train_standardized[X_train.columns.difference(encoded_columns)] = min_max_scaler.fit_transform(
        X_train_standardized[X_train.columns.difference(encoded_columns)])
    X_test_standardized[X_train.columns.difference(encoded_columns)] = min_max_scaler.transform(
        X_test_standardized[X_train.columns.difference(encoded_columns)])

    # target class balancing
    smote = SMOTE(random_state=5805)
    y_train_df = pd.DataFrame(Y_train, columns=['RainTomorrow'])
    sns.countplot(data=y_train_df, x='RainTomorrow')
    plt.show()

    X_train_standardized_knn, Y_train_knn = smote.fit_resample(X_train_standardized, Y_train)
    y_train_df = pd.DataFrame(Y_train_knn, columns=['RainTomorrow'])
    sns.countplot(data=y_train_df, x='RainTomorrow')
    plt.show()

    ##---------------
    ## RFA
    ##----------------

    rf_model = RandomForestRegressor(n_estimators=100, random_state=5805)
    rf_model.fit(X_train_standardized, Y_train)
    feature_importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X_train_standardized.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    selected_feature_importance_df = feature_importance_df[feature_importance_df['Importance'] >= 0.01]

    plt.figure(figsize=(10, 6))
    plt.barh(selected_feature_importance_df['Feature'], selected_feature_importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance in Random Forest Analysis (Selected Features)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    print("Selected features after RFA with threshold 0.01 are: ", selected_feature_importance_df['Feature'])
    print("# of features after RFA with threshold 0.01 are: ", len(selected_feature_importance_df))

    # pca
    condition_number_before_pca = np.linalg.cond(X_train_standardized)
    U, singular_values_before_pca, Vt = np.linalg.svd(X_train_standardized, full_matrices=False)

    print("Singular values before pca:", np.round(singular_values_before_pca, 3))
    print("Condition number before PCA: ", round(condition_number_before_pca, 3))
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_standardized)
    X_test_pca = pca.transform(X_test_standardized)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    var_95 = np.argmax(cumulative_variance_ratio > 0.95) + 1
    X_test_pca = X_test_pca[:, :var_95]
    X_train_pca = X_train_pca[:, :var_95]

    print("--------Features needed that explain more than 95% of the dependent variance: ", var_95)

    num_features = len(explained_variance_ratio)
    x = np.arange(1, num_features + 1)  # Number of features
    plt.figure()
    plt.plot(x, cumulative_variance_ratio, linestyle='-', color='b')
    plt.xlabel('Number of Features/Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance Ratio vs. Number of Features')
    plt.axhline(y=0.95, color='red', linestyle='--', label='95% Threshold')
    plt.axvline(x=var_95, color='green', linestyle='--', label=f'# components for 95% var:{var_95}')
    plt.grid(True)
    plt.legend()
    plt.show()
    pca = PCA(n_components=var_95)
    reduced_data = pca.fit_transform(X_train_standardized)
    condition_number_after_pca = np.linalg.cond(reduced_data)
    U, singular_values_after_pca, Vt = np.linalg.svd(reduced_data, full_matrices=False)

    print("Singular values after pca:", np.round(singular_values_after_pca, 3))
    print("Condition number after PCA: ", round(condition_number_after_pca, 3))

    selected_columns = selected_feature_importance_df['Feature'].tolist()

    # plotting heatmaps
    # Calculate the covariance matrix
    cov_matrix = X_train_standardized[selected_columns].cov()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Sample Covariance Matrix Heatmap')
    plt.tight_layout()
    plt.show()

    corr_matrix = X_train_standardized[selected_columns].corr()

    # Create a heatmap of the correlation coefficients matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Sample Pearson Correlation Coefficients Heatmap')
    plt.tight_layout()
    plt.show()

    # removing correlation
    selected_features_rfa = selected_feature_importance_df['Feature'].tolist()
    selected_features_rfa = [feature for feature in selected_features_rfa if
                             feature not in ['Pressure9am', 'Temp9am', 'MaxTemp', 'HumidityTemp9am', 'Evaporation']]

    # plotting correlation heatmap again after removing correlated features
    corr_matrix = X_train_standardized[selected_features_rfa].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Sample Pearson Correlation Coefficients Heatmap-after removing highly correlated features')
    plt.tight_layout()
    plt.show()

    def plot_multiple_roc_curves(y_test, y_probas, model_labels):
        fig = go.Figure()

        for i in range(len(y_probas)):
            fpr, tpr, _ = roc_curve(y_test, y_probas[i])
            auc = roc_auc_score(y_test, y_probas[i])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_labels[i]} AUC = {auc:.3f}'))

        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline', line=dict(dash='dash', color='red')))

        fig.update_layout(
            title='ROC Curves for Multiple Models',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=1000,
            height=800,
            title_x=0.5,

        )
        fig.show()

    # %%
    def calc_metrics(Y_test, y_pred, model):
        print(f"Calculating metrics for model {model} : ")
        accuracy = accuracy_score(Y_test, y_pred)
        print(f'Accuracy : {accuracy:.2f}')
        conf_matrix = confusion_matrix(Y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        disp.plot()
        plt.title(f"Confusion Matrix for Model {model}")
        plt.show()
        print("Confusion Matrix:")
        print(conf_matrix)
        # Calculate precision
        precision = precision_score(Y_test, y_pred)
        print(f"Precision: {precision:.2f}")
        # Calculate recall (sensitivity)
        recall = recall_score(Y_test, y_pred)
        print(f"Recall (Sensitivity): {recall:.2f}")
        # Calculate specificity
        specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        print(f"Specificity: {specificity:.2f}")
        # Calculate F1-score
        f1 = f1_score(Y_test, y_pred)
        print(f"F1-Score: {f1:.2f}")

        metrics = {
            'Model': model,
            'Accuracy': round(accuracy, 2),
            'Precision': round(precision, 2),
            'Recall': round(recall, 2),
            'F1-Score': round(f1, 2),
            'Specificity': round(specificity, 2),
            'Confusion matrix': conf_matrix
        }

        all_model_metrics.append(metrics)

    def cross_validate(model, X_train_selected, Y_train):
        k = 5
        stratified_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=5805)
        # Perform Stratified k-fold cross-validation and obtain accuracy scores
        accuracy_scores = cross_val_score(model, X_train_selected, Y_train, cv=stratified_kfold,
                                          scoring='accuracy')
        # Print the accuracy scores for each fold
        for fold, accuracy in enumerate(accuracy_scores, 1):
            print(f'Fold {fold}: Accuracy = {accuracy:.2f}')
        mean_accuracy = accuracy_scores.mean()
        print(f'Mean Accuracy: {mean_accuracy:.2f}')

    def roc_auc(Y_test, y_proba, model):
        logistic_fpr, logistic_tpr, _ = roc_curve(Y_test, y_proba)
        auc_logstic = roc_auc_score(Y_test, y_proba)
        print(f'{model} auc = {auc_logstic:.3f}')
        plt.figure()
        plt.plot(logistic_fpr, logistic_tpr, label=f'{model} auc = {auc_logstic:.3f}')
        plt.plot(logistic_fpr, logistic_fpr, 'r--')
        plt.legend(loc=4)
        plt.grid()
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'{model} roc curve')
        plt.show()

    def grid_search_logit(X_train_selected, Y_train):
        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10],
            'solver': ['liblinear'],
            'max_iter': [100, 300, 500],
            'class_weight': [{0: 0.1, 1: 0.9}, {0: 0.3, 1: 0.7}]
        }
        logistic_model = LogisticRegression(random_state=5805)
        grid_search = GridSearchCV(logistic_model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train_selected, Y_train)
        return grid_search.best_params_

    def display_metrics_prettytable(metrics_list):
        table = PrettyTable()
        table.field_names = list(metrics_list[0].keys())
        for metrics in metrics_list:
            table.add_row([metrics[key] for key in table.field_names])
        table.title = 'Metrics List for all models'
        print(table)

    # # --------------------
    # # Logistic Regression
    # # --------------------
    print("---------------Performing Logistic regression---------------")
    Y_train_pca = Y_train
    X_train_selected = X_train_standardized[selected_features_rfa]

    best_params_rfa_logit = grid_search_logit(X_train_selected, Y_train)
    print("Best params logit: ", best_params_rfa_logit)

    logistic_model = LogisticRegression(random_state=5805, **best_params_rfa_logit)
    logistic_model.fit(X_train_selected, Y_train)
    y_pred_logistic_rfa = logistic_model.predict(X_test_standardized[selected_features_rfa])

    y_proba_logistic_rfa = logistic_model.predict_proba(X_test_standardized[selected_features_rfa])[::, -1]
    cross_validate(logistic_model, X_train_selected, Y_train)
    calc_metrics(Y_test, y_pred_logistic_rfa, 'Logistic Regression')
    roc_auc(Y_test, y_proba_logistic_rfa, 'Logistic Regression')
    all_probabilities.append(y_proba_logistic_rfa)
    model_labels.append('Logistic Regression')

    # logit for pca
    best_params_pca_logit = grid_search_logit(X_train_pca, Y_train_pca)
    print("Best params logit: ", best_params_pca_logit)
    logistic_model = LogisticRegression(random_state=5805, **best_params_pca_logit)
    logistic_model.fit(X_train_pca, Y_train_pca)
    y_pred_logistic_pca = logistic_model.predict(X_test_pca)
    y_pred_train_logistic_pca = logistic_model.predict(X_train_pca)
    y_proba_logistic_pca = logistic_model.predict_proba(X_test_pca)[::, -1]
    cross_validate(logistic_model, X_train_pca, Y_train_pca)
    calc_metrics(Y_test, y_pred_logistic_pca, 'Logistic regression using PCA')
    roc_auc(Y_test, y_proba_logistic_pca, 'Logistic regression using PCA')
    # ---------------
    # decision trees
    # ---------------
    print("---------------Performing Decision tree---------------")
    custom_weights = {0: 3, 1: 7}
    model_trees = ExtraTreesClassifier(random_state=5805, class_weight=custom_weights)
    model_trees.fit(X_train_selected, Y_train)
    print("Feature importances: \n", np.round(model_trees.feature_importances_, 2))

    clf = DecisionTreeClassifier(random_state=5805, class_weight=custom_weights)
    clf.fit(X_train_selected, Y_train)
    y_train_predicted_dt = clf.predict(X_train_selected)
    y_test_predicted_dt = clf.predict(X_test_standardized[selected_features_rfa])
    print(f'Train accuracy of full tree {round(accuracy_score(Y_train, y_train_predicted_dt), 2)}')
    print(f'Test accuracy of full tree {round(accuracy_score(Y_test, y_test_predicted_dt), 2)}')
    cross_validate(clf, X_train_selected, Y_train)
    calc_metrics(Y_test, y_test_predicted_dt, 'Decision Tree')
    roc_auc(Y_test, clf.predict_proba(X_test_standardized[selected_features_rfa])[::, 1], 'Decision Tree')
    all_probabilities.append(clf.predict_proba(X_test_standardized[selected_features_rfa])[::, 1])
    model_labels.append('Decision Tree')

    # Pre pruning
    print("---------------Finding Pre pruned decision tree---------------")
    tuned_parameters = {
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 6, 8, 10, 20],
        'min_samples_leaf': [5, 10, 15],
        'max_features': ['sqrt', 'log2'],
        'splitter': ['best', 'random'],
        'criterion': ['gini', 'entropy'],
        'class_weight': [{0: 3, 1: 7}, {0: 4, 1: 6}, {0: 1, 1: 9}]
    }
    clf1 = DecisionTreeClassifier(random_state=5805)
    grid_search_pre_pruned = GridSearchCV(clf1, tuned_parameters, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_pre_pruned.fit(X_train_selected, Y_train)
    best_params_DT = grid_search_pre_pruned.best_params_
    best_estimator = grid_search_pre_pruned.best_estimator_
    print("Best Parameters for prePruning: ", best_params_DT)

    clf_pre = DecisionTreeClassifier(**best_params_DT, random_state=5805)
    clf_pre.fit(X_train_selected, Y_train)
    y_train_predicted_pre = clf_pre.predict(X_train_selected)
    y_test_predicted_pre = clf_pre.predict(X_test_standardized[selected_features_rfa])
    print(f'Train accuracy of pre Pruned Tree {round(accuracy_score(Y_train, y_train_predicted_pre), 2)}')
    print(f'Test accuracy of pre Pruned Tree {round(accuracy_score(Y_test, y_test_predicted_pre), 2)}')
    plt.figure(figsize=(20, 12))
    tree.plot_tree(clf_pre, rounded=True, filled=True)
    plt.show()
    cross_validate(clf_pre, X_train_selected, Y_train)
    calc_metrics(Y_test, y_test_predicted_pre, 'Pre-pruned Tree')
    roc_auc(Y_test, clf_pre.predict_proba(X_test_standardized[selected_features_rfa])[::, 1], 'Pre-pruned Tree')
    all_probabilities.append(clf_pre.predict_proba(X_test_standardized[selected_features_rfa])[::, 1])
    model_labels.append('Pre-pruned Decision Tree')

    # post pruning
    print("---------------Finding Post pruned decision tree---------------")
    clf2 = DecisionTreeClassifier(random_state=5805, class_weight={0: 4, 1: 6})
    path = clf2.cost_complexity_pruning_path(X_train_selected, Y_train)
    alphas = path['ccp_alphas']
    accuracy_train, accuracy_test = [], []
    for i in alphas:
        clf2 = DecisionTreeClassifier(ccp_alpha=i, random_state=5805, class_weight={0: 4, 1: 6})
        clf2.fit(X_train_selected, Y_train)
        y_train_pred = clf2.predict(X_train_selected)
        y_test_pred = clf2.predict(X_test_standardized[selected_features_rfa])
        accuracy_train.append(accuracy_score(Y_train, y_train_pred))
        accuracy_test.append(accuracy_score(Y_test, y_test_pred))
    plt.figure()
    plt.xlabel('alpha')
    plt.ylabel('accuracy')
    plt.title("Accuracy vs alpha for training and testing sets")
    plt.plot(alphas[:500], accuracy_train[:500], marker="o", label="train",
             drawstyle="steps-post")
    plt.plot(alphas[:500], accuracy_test[:500], marker="o", label="test",
             drawstyle="steps-post")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    optimum_alpha = alphas[accuracy_test.index(max(accuracy_test))]
    print("Optimum alpha: ", optimum_alpha)
    clf_post = DecisionTreeClassifier(random_state=5805, ccp_alpha=optimum_alpha, class_weight={0: 4, 1: 6})
    clf_post.fit(X_train_selected, Y_train)
    y_train_pred_post = clf_post.predict(X_train_selected)
    y_test_pred_post = clf_post.predict(X_test_standardized[selected_features_rfa])
    print(f'Train accuracy of post Pruned Tree {accuracy_score(Y_train, y_train_pred_post):.2f}')
    print(f'Test accuracy of Post Pruned Tree {accuracy_score(Y_test, y_test_pred_post):.2f}')
    plt.figure(figsize=(16, 8))
    tree.plot_tree(clf_post, rounded=True, filled=True)
    plt.show()
    cross_validate(clf_post, X_train_selected, Y_train)
    calc_metrics(Y_test, y_test_pred_post, 'Post-Pruned Tree')
    all_probabilities.append(clf_post.predict_proba(X_test_standardized[selected_features_rfa])[::, 1])
    model_labels.append('Post-pruned Decision Tree')
    roc_auc(Y_test, clf_post.predict_proba(X_test_standardized[selected_features_rfa])[::, 1], 'Post-Pruned Tree')

    # # KNN
    print("---------------Performing KNN---------------")
    error_rate = []
    k_range = range(1, 20, 2)  # Adjust the range based on your dataset and needs

    for i in k_range:
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train_standardized_knn[selected_features_rfa], Y_train_knn)
        y_pred_i = knn.predict(X_test_standardized[selected_features_rfa].values)
        error_rate.append(np.mean(y_pred_i != Y_test))
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red')
    plt.xticks(k_range)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('Number of Neighbors: K')
    plt.ylabel('Error Rate')
    plt.show()

    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(X_train_standardized_knn[selected_features_rfa], Y_train_knn)
    x_test_array = X_test_standardized[selected_features_rfa].values
    y_pred_knn = knn.predict(x_test_array)
    calc_metrics(Y_test, y_pred_knn, 'KNN')
    cross_validate(knn, X_train_standardized_knn[selected_features_rfa].values, Y_train_knn)
    knn_proba = knn.predict_proba(X_test_standardized[selected_features_rfa].values)[::, 1]
    roc_auc(Y_test, knn_proba, 'KNN')
    all_probabilities.append(knn_proba)
    model_labels.append('KNN')

    # %%
    # Neural networks
    print("---------------Neural Network---------------")

    def create_model(input_dim):
        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=LegacyAdam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def cross_validate_nn(X, Y, n_splits=5, epochs=50, batch_size=32):
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
        fold_no = 1
        accuracies = []

        for train, test in kfold.split(X, Y):
            model = create_model(input_dim=X.shape[1])

            history = model.fit(X[train], Y[train], epochs=epochs, batch_size=batch_size, verbose=0,
                                validation_split=0.2)

            scores = model.evaluate(X[test], Y[test], verbose=0)
            print(f'Fold {fold_no}: Accuracy = {scores[1]:.2f}')
            accuracies.append(scores[1])

            fold_no += 1

        mean_accuracy = np.mean(accuracies)
        print(f'Mean Accuracy: {mean_accuracy:.2f}')

    X = X_train_standardized[selected_features_rfa].values
    Y = Y_train.values

    cross_validate_nn(X, Y)

    model_ann = create_model(X_train_standardized[selected_features_rfa].values.shape[1])
    history = model_ann.fit(X_train_standardized[selected_features_rfa].values, Y_train,
                            epochs=50, batch_size=32, validation_split=0.2)

    x_test_array = X_test_standardized[selected_features_rfa].values
    y_test_array = Y_test.values
    loss, accuracy = model_ann.evaluate(x_test_array, y_test_array)
    print(f'Test loss: {loss:.2f}, Test accuracy: {accuracy:.2f}')

    plt.plot(history.history['accuracy'], label='Accuracy (training data)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Loss (training data)')
    plt.plot(history.history['val_loss'], label='Loss (validation data)')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    y_pred_probabilities_ann = model_ann.predict(x_test_array)
    y_pred_labels_ann = (y_pred_probabilities_ann > 0.4).astype("int32")
    calc_metrics(Y_test, y_pred_labels_ann, 'Neural Network')
    roc_auc(Y_test, y_pred_probabilities_ann, 'Neural Network')
    all_probabilities.append(y_pred_probabilities_ann)
    model_labels.append('Neural Network')

    # Random forest- Bagging
    print("---------------Performing random forest- Bagging---------------")

    classes = np.unique(Y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=Y_train)

    class_weight_dict = dict(zip(classes, class_weights))

    parameter_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    rf = RandomForestClassifier(random_state=5805, class_weight=class_weight_dict)

    grid_search_rf = GridSearchCV(estimator=rf, param_grid=parameter_grid,
                                  cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search_rf.fit(X_train_standardized[selected_features_rfa], Y_train)
    print("Best Parameters:", grid_search_rf.best_params_)
    print("Best Score:", grid_search_rf.best_score_)
    best_rf_model = grid_search_rf.best_estimator_

    y_pred_rf = best_rf_model.predict(X_test_standardized[selected_features_rfa])

    calc_metrics(Y_test, y_pred_rf, 'Random Forest Bagging')

    probabilities_rf = best_rf_model.predict_proba(X_test_standardized[selected_features_rfa])[:, 1]
    roc_auc(Y_test, probabilities_rf, 'Randon Forest-Bagging')

    all_probabilities.append(probabilities_rf)
    model_labels.append('Random Forest- Bagging')

    cross_validate(best_rf_model, X_train_standardized[selected_features_rfa], Y_train)

    # Boosting
    print("---------------Performing Boosting---------------")
    class_counts = np.bincount(Y_train)
    scale_pos_weight = class_counts[0] / class_counts[1]

    parameter_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 10, 15],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }

    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                  random_state=5805, scale_pos_weight=scale_pos_weight)

    grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=parameter_grid,
                                   cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search_xgb.fit(X_train_standardized[selected_features_rfa], Y_train)
    print("Best Parameters:", grid_search_xgb.best_params_)
    print("Best Score:", grid_search_xgb.best_score_)
    best_xgb_model = grid_search_xgb.best_estimator_

    y_pred_xgb = best_xgb_model.predict(X_test_standardized[selected_features_rfa])

    calc_metrics(Y_test, y_pred_xgb, 'XGB Boosting')

    probabilities_xgb = best_xgb_model.predict_proba(X_test_standardized[selected_features_rfa])[:, 1]
    roc_auc(Y_test, probabilities_xgb, 'XGB Boosting')

    all_probabilities.append(probabilities_xgb)
    model_labels.append('XGB Boosting')

    cross_validate(best_xgb_model, X_train_standardized[selected_features_rfa],
                   Y_train)

    # SVM

    print("---------------Performing SVM---------------")

    parameter_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    grid_svm_model = SVC(probability=True, class_weight=class_weight_dict)

    grid_search_svm = GridSearchCV(estimator=grid_svm_model, param_grid=parameter_grid,
                                   cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search_svm.fit(X_train_selected, Y_train)
    print("Best Parameters:", grid_search_svm.best_params_)
    print("Best Score:", grid_search_svm.best_score_)

    svm_model = grid_search_svm.best_estimator_
    y_pred_svm = svm_model.predict(X_test_standardized[selected_features_rfa])
    y_pred_train_svm = svm_model.predict(X_train_standardized[selected_features_rfa])
    y_proba_svm = svm_model.predict_proba(X_test_standardized[selected_features_rfa])[::, -1]

    cross_validate(svm_model, X_train_selected, Y_train)

    calc_metrics(Y_test, y_pred_svm, 'SVM')
    roc_auc(Y_test, y_proba_svm, 'SVM')
    all_probabilities.append(y_proba_svm)
    model_labels.append('SVM')

    # kmeans
    print("------------------Performing Kmeans--------------------")

    range_n_clusters = list(range(2, 10))
    silhouette_avg_scores = []
    wcss = []

    for num_clusters in range_n_clusters:
        # Initialize KMeans with k clusters
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=5805)
        cluster_labels = kmeans.fit_predict(X_train_standardized_knn[selected_features_rfa])
        wcss.append(kmeans.inertia_)
        silhouette_avg = silhouette_score(X_train_standardized_knn[selected_features_rfa], cluster_labels)
        silhouette_avg_scores.append(silhouette_avg)
        print(f"For n_clusters={num_clusters}, the silhouette score is {silhouette_avg}")

    plt.plot(range_n_clusters, silhouette_avg_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette Analysis For Optimal k')
    plt.show()

    plt.plot(range_n_clusters, wcss, 'bo-')
    plt.title('Within-Cluster Variation Plot')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.show()

    kmeans_optimal = KMeans(n_clusters=4, init='k-means++', random_state=5808)
    kmeans_optimal.fit(X_train_standardized_knn[selected_features_rfa])
    y_pred_kmeans = kmeans_optimal.predict(X_test_standardized[selected_features_rfa])
    sil_score = silhouette_score(X_test_standardized[selected_features_rfa], y_pred_kmeans)
    print(f"silhouette score after kmeans: {sil_score:.3f}")

    plot_multiple_roc_curves(Y_test, all_probabilities, model_labels)

    display_metrics_prettytable(all_model_metrics)


def linearRegression():
    df = pd.read_csv('weatherAUS.csv')
    print((df.isna().sum()) / len(df) * 100)
    sunshine_median = df['Sunshine'].median()
    evaporation_median = df['Evaporation'].median()
    cloud9am_mode = df['Cloud9am'].mode()
    cloud3pm_mode = df['Cloud3pm'].mode()
    df['Sunshine'] = df['Sunshine'].fillna(sunshine_median)
    df['Evaporation'] = df['Evaporation'].fillna(evaporation_median)
    df['Cloud9am'] = df['Cloud9am'].fillna(cloud9am_mode[0])
    df['Cloud3pm'] = df['Cloud3pm'].fillna(cloud3pm_mode[0])

    print("after dropping columns that have high values null values:\n", (df.isna().sum()) / len(df) * 100)

    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = numerical_cols.drop(['Rainfall', 'Cloud9am', 'Cloud3pm'])

    for col in numerical_cols:
        median = df[col].median()
        df[col].fillna(median, inplace=True)
    mode_temp = df['WindGustDir'].mode()
    df['WindGustDir'] = df['WindGustDir'].fillna(mode_temp[0])
    mode_temp = df['WindDir9am'].mode()
    df['WindDir9am'] = df['WindDir9am'].fillna(mode_temp[0])
    mode_temp = df['WindDir3pm'].mode()
    df['WindDir3pm'] = df['WindDir3pm'].fillna(mode_temp[0])
    df.dropna(inplace=True)
    print("after cleaning rows that have null values:\n", (df.isnull().sum()) / len(df) * 100)
    cleaned_df = df.copy()
    df.drop(inplace=True, columns=['Date'])


    # remove outliers

    def remove_outliers(dataframe, columns):
        import numpy as np
        clean_data = dataframe.copy()
        for column in columns:
            q1 = np.percentile(clean_data[column], 25)
            q3 = np.percentile(clean_data[column], 75)
            iqr = q3 - q1

            lower_bound = q1 - 4.5 * iqr
            upper_bound = q3 + 4.5 * iqr

            print(f"Q1 and Q3 for {column}: {q1:.2f}, {q3:.2f}.")
            print(f"IQR for {column}: {iqr:.2f}.")
            print(f"Values below {lower_bound:.2f} or above {upper_bound:.2f} are outliers for {column}.")

            # Removing outliers
            clean_data = clean_data[(clean_data[column] >= lower_bound) & (clean_data[column] <= upper_bound)]

        return clean_data

    numeric_columns = ['Rainfall', 'MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am',
                       'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']

    df = remove_outliers(df, numeric_columns)
    # encoding
    encoding_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
    df_resampled1 = df.drop(columns=encoding_columns)
    df_resampled1.drop(columns=['RainTomorrow', 'RainToday'], inplace=True)
    df_encoded = df[['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']]
    df_encoded = pd.get_dummies(df_encoded, columns=encoding_columns, drop_first=True)
    df_encoded['RainTomorrow'] = df_encoded['RainTomorrow'].map({'No': 0, 'Yes': 1})
    df_encoded['RainToday'] = df_encoded['RainToday'].map({'No': 0, 'Yes': 1})
    df_encoded = df_encoded.astype(int)
    encoded_columns = [col for col in df_encoded.columns]
    df_resampled1 = pd.concat([df_resampled1, df_encoded], axis=1)
    Y = df_resampled1['Humidity3pm']
    df_resampled1.drop(columns=['Humidity3pm'], inplace=True)
    df_resampled1.drop(columns=['Humidity9am'], inplace=True)
    df_resampled1.drop(columns=['RainTomorrow'], inplace=True)

    X_train, X_test, Y_train, Y_test = train_test_split(df_resampled1, Y,
                                                        shuffle=True, test_size=0.2, random_state=5805)

    X_train_standardized = X_train.copy()
    X_test_standardized = X_test.copy()
    x_standard_scaler = StandardScaler()
    y_standard_scaler = StandardScaler()

    # Standardize the data
    X_train_standardized[X_train.columns.difference(encoded_columns)] = x_standard_scaler.fit_transform(
        X_train_standardized[X_train.columns.difference(encoded_columns)])
    X_test_standardized[X_train.columns.difference(encoded_columns)] = x_standard_scaler.transform(
        X_test_standardized[X_train.columns.difference(encoded_columns)])

    Y_train_standardized = y_standard_scaler.fit_transform(Y_train.values.reshape(-1, 1))
    Y_test_standardized = y_standard_scaler.transform(Y_test.values.reshape(-1, 1))

    Y_train_standardized = pd.Series(Y_train_standardized.flatten(), index=Y_train.index)
    Y_test_standardized = pd.Series(Y_test_standardized.flatten(), index=Y_test.index)

    # RFA

    rf_model = RandomForestRegressor(n_estimators=100, random_state=5805)
    rf_model.fit(X_train_standardized, Y_train_standardized)
    feature_importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X_train_standardized.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    selected_feature_importance_df = feature_importance_df[feature_importance_df['Importance'] >= 0.01]

    plt.figure(figsize=(10, 6))
    plt.barh(selected_feature_importance_df['Feature'], selected_feature_importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance in Random Forest Analysis for linear regression (Selected Features)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    print("Selected features after RFA with threshold 0.01 are: ", selected_feature_importance_df['Feature'])
    print("# of features after RFA with threshold 0.01 are: ", len(selected_feature_importance_df))

    # PCA

    from sklearn.decomposition import PCA
    import numpy as np
    condition_number_before_pca = np.linalg.cond(X_train_standardized)
    U, singular_values_before_pca, Vt = np.linalg.svd(X_train_standardized, full_matrices=False)

    print("Singular values before pca:", np.round(singular_values_before_pca, 3))
    print("Condition number before PCA: ", round(condition_number_before_pca, 3))
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_standardized)
    X_test_pca = pca.transform(X_test_standardized)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    var_95 = np.argmax(cumulative_variance_ratio > 0.95) + 1
    X_test_pca = X_test_pca[:, :var_95]
    X_train_pca = X_train_pca[:, :var_95]

    print("--------Features needed that explain more than 95% of the dependent variance: ", var_95)

    num_features = len(explained_variance_ratio)
    x = np.arange(1, num_features + 1)  # Number of features
    plt.figure()
    plt.plot(x, cumulative_variance_ratio, linestyle='-', color='b')
    plt.xlabel('Number of Features/Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance Ratio vs. Number of Features')
    plt.axhline(y=0.95, color='red', linestyle='--', label='95% Threshold')
    plt.axvline(x=var_95, color='green', linestyle='--', label=f'# components for 95% var:{var_95}')
    plt.grid(True)
    plt.legend()
    plt.show()

    pca = PCA(n_components=var_95)
    reduced_data = pca.fit_transform(X_train_standardized)
    condition_number_after_pca = np.linalg.cond(reduced_data)
    U, singular_values_after_pca, Vt = np.linalg.svd(reduced_data, full_matrices=False)

    print("Singular values after pca:", np.round(singular_values_after_pca, 3))
    print("Condition number after PCA: ", round(condition_number_after_pca, 3))

    comparison_table = PrettyTable()
    comparison_table.field_names = ["Model", "R-squared", "Adj. R-squared", "AIC", "BIC", "MSE"]

    # linear regression for RFA and PCA feature selected features
    from statsmodels.api import OLS
    import statsmodels.api as sm
    from sklearn.metrics import mean_squared_error
    # for rfa
    X_train_with_constant = sm.add_constant(X_train_standardized[selected_feature_importance_df['Feature'].tolist()])

    # Fit OLS model
    model = OLS(Y_train_standardized, X_train_with_constant).fit()

    # Getting the summary of the regression
    model_summary = model.summary()

    print(model_summary)
    r_squared_rfa = model.rsquared
    adj_r_squared_rfa = model.rsquared_adj
    aic_rfa = model.aic
    bic_rfa = model.bic

    X_test_with_constant = sm.add_constant(
        X_test[selected_feature_importance_df['Feature'].tolist()])

    Y_pred = model.predict(X_test_with_constant)
    mse = mean_squared_error(Y_test_standardized, Y_pred)
    print(mse)
    comparison_table.add_row([
        "RFA",
        round(r_squared_rfa, 3),
        round(adj_r_squared_rfa, 3),
        round(aic_rfa, 3),
        round(bic_rfa, 3),
        round(mse, 3)
    ])

    # PCA
    X_train_with_constant = sm.add_constant(reduced_data)

    # Fit OLS model
    model = OLS(Y_train_standardized, X_train_with_constant).fit()

    # Getting the summary of the regression
    model_summary = model.summary()

    print(model_summary)
    r_squared_pca = model.rsquared
    adj_r_squared_pca = model.rsquared_adj
    aic_pca = model.aic
    bic_pca = model.bic

    from sklearn.metrics import mean_squared_error
    X_test_pca = pca.transform(X_test_standardized)
    X_test_with_constant = sm.add_constant(X_test_pca)

    Y_pred = model.predict(X_test_with_constant)
    mse = mean_squared_error(Y_test_standardized, Y_pred)
    print(mse)
    comparison_table.add_row([
        "PCA",
        round(r_squared_pca, 3),
        round(adj_r_squared_pca, 3),
        round(aic_pca, 3),
        round(bic_pca, 3),
        round(mse, 3)
    ])

    # Backward stepwise regression

    df_vif = X_train_standardized.copy()

    vif = pd.DataFrame()
    vif["Variable"] = df_vif.columns
    vif["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
    print("VIF before removal:\n ")

    print(vif)

    df_vif.drop(columns=['MaxTemp'], inplace=True, axis=1)

    df_vif.drop(columns=['Pressure9am'], inplace=True, axis=1)

    df_vif.drop(columns=['Temp9am'], inplace=True, axis=1)

    df_vif.drop(columns=['RainToday'], inplace=True, axis=1)

    vif = pd.DataFrame()
    vif["Variable"] = df_vif.columns
    vif["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
    selected_columns = df_vif.columns
    print("vif after removal: \n")
    print(vif)

    # after removing collinearity
    X_back = X_train_standardized[selected_columns].copy()
    X_train_with_constant = sm.add_constant(X_back)

    # Fit OLS model
    model = OLS(Y_train_standardized, X_train_with_constant).fit()

    model_summary = model.summary()

    print(model_summary)

    stats_list = []

    # List of columns to drop in each iteration
    columns_to_drop = [
        'WindDir9am_ENE', 'WindGustDir_SSE', 'WindGustDir_SE', 'WindGustDir_ENE',
        'WindDir9am_NE', 'WindDir3pm_N', 'WindGustDir_NW', 'WindGustDir_SSW',
        'Location_Woomera', 'WindDir3pm_ESE', 'WindDir3pm_NNW', 'WindGustDir_S',
        'Location_Uluru', 'WindDir3pm_S', 'WindDir3pm_SSE', 'WindDir3pm_SE', 'WindGustDir_ESE',
        'WindDir3pm_SSW', 'WindDir9am_N', 'WindDir9am_NNW', 'WindDir9am_NNE'
    ]

    for col in columns_to_drop:
        # Add constant to the predictor variables
        X_train_with_constant = sm.add_constant(X_back)

        # Fit the OLS model
        model = OLS(Y_train_standardized, X_train_with_constant).fit()

        # Extract the required statistics
        r_squared = round(model.rsquared, 3)
        adj_r_squared = round(model.rsquared_adj, 3)
        aic = round(model.aic, 3)
        bic = round(model.bic, 3)
        f_statistic = round(model.fvalue, 3)
        f_pvalue = round(model.f_pvalue, 3)
        t_statistic = round(model.tvalues[col], 3)
        t_pvalue = round(model.pvalues[col], 3)

        stats_tuple = (col, r_squared, adj_r_squared, aic, bic, f_statistic, f_pvalue, t_statistic, t_pvalue)
        stats_list.append(stats_tuple)

        # Drop the specified column
        X_back.drop(columns=[col], inplace=True)

    # Creating a PrettyTable to display the statistics
    stats_table = PrettyTable()
    stats_table.field_names = ["Dropped Variable", "R-squared", "Adj. R-squared", "AIC", "BIC", "F-statistic",
                               "F-pvalue", "T-statistic", "T-pvalue"]

    for stat in stats_list:
        stats_table.add_row(stat)

    print(stats_table)

    X_train_with_constant = sm.add_constant(X_back)
    model = OLS(Y_train_standardized, X_train_with_constant).fit()

    model_summary = model.summary()

    print(model_summary)

    r_squared_backward = model.rsquared
    adj_r_squared_backward = model.rsquared_adj
    aic_backward = model.aic
    bic_backward = model.bic

    from sklearn.metrics import mean_squared_error
    X_test_with_constant = sm.add_constant(
        X_test_standardized[X_back.columns])  # Add a constant column for the intercept

    # Predict using the OLS model
    Y_pred = model.predict(X_test_with_constant)
    mse = mean_squared_error(Y_test_standardized, Y_pred)
    print(mse)

    comparison_table.add_row([
        "Backward stepwise",
        round(r_squared_backward, 3),
        round(adj_r_squared_backward, 3),
        round(aic_backward, 3),
        round(bic_backward, 3),
        round(mse, 3)
    ])

    print(comparison_table)

    SSR = np.sum((Y_test_standardized - Y_pred) ** 2)

    SST = np.sum((Y_test_standardized - np.mean(Y_test_standardized)) ** 2)

    R_squared = 1 - (SSR / SST)
    print(f"R-squared on Test Data: {R_squared}")

    # confidence interval

    predictions = model.get_prediction(X_test_with_constant)

    prediction_intervals = predictions.summary_frame(alpha=0.05)
    predictions_original = y_standard_scaler.inverse_transform(
        prediction_intervals['mean'].values.reshape(-1, 1)).ravel()
    lower_bound_original = y_standard_scaler.inverse_transform(
        prediction_intervals['obs_ci_lower'].values.reshape(-1, 1)).ravel()
    upper_bound_original = y_standard_scaler.inverse_transform(
        prediction_intervals['obs_ci_upper'].values.reshape(-1, 1)).ravel()

    x_seq = list(range(len(predictions_original)))
    plt.figure(figsize=(10, 6))
    plt.plot(x_seq, predictions_original, color='blue', label='Predicted Humidity at 3pm')
    plt.fill_between(x_seq, lower_bound_original, upper_bound_original, color='grey', alpha=0.5,
                     label='95% Prediction Interval')
    plt.xlabel('Observations')
    plt.ylabel('Humidity')
    plt.title('Predicted Humidity and 95% Prediction Interval')
    plt.legend()
    plt.tight_layout()
    plt.show()

    num_observations = 100

    predictions_original_subset = predictions_original[:num_observations]
    lower_bound_original_subset = lower_bound_original[:num_observations]
    upper_bound_original_subset = upper_bound_original[:num_observations]
    x_seq_subset = x_seq[:num_observations]
#%%
    plt.figure(figsize=(10, 6))
    plt.plot(x_seq_subset, predictions_original_subset, color='blue', label='Predicted Humidity at 3pm')
    plt.fill_between(x_seq_subset, lower_bound_original_subset, upper_bound_original_subset, color='grey', alpha=0.5,
                     label='95% Prediction Interval')
    plt.xlabel('Observations')
    plt.ylabel('Humidity')
    plt.title('Predicted Humidity and 95% Prediction Interval for First 100 Observations')
    plt.legend()
    plt.tight_layout()
    plt.show()
#%%
    # plot y pred and y test
    y_test_original = y_standard_scaler.inverse_transform(Y_test_standardized.values.reshape([-1, 1])).ravel()
    y_predictions_original = y_standard_scaler.inverse_transform(Y_pred.values.reshape([-1, 1])).ravel()
    plt.figure()
    plt.plot(y_test_original, label='Original Humidity', color='blue')
    plt.plot(y_predictions_original, label='Predicted Humidity', color='red', linestyle='dashed')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Humidity')
    plt.title(f'Original vs Predicted Humidity')
    plt.show()

    num_observations = 100

    y_test_original_subset = y_test_original[:num_observations]
    y_predictions_original_subset = y_predictions_original[:num_observations]

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_original_subset, label='Original Humidity', color='blue')
    plt.plot(y_predictions_original_subset, label='Predicted Humidity', color='red', linestyle='dashed')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Humidity')
    plt.title('Original vs Predicted Humidity for First 100 Observations')
    plt.show()


def main():
    # Call the Classification function
    print("Running Classification Analysis...")
    Classification()

    # Call the LinearRegression function
    print("\nRunning Linear Regression Analysis...")
    linearRegression()


# Execute the main function
if __name__ == "__main__":
    main()
