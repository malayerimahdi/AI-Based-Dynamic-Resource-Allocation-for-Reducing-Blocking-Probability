# **Step 1: Load the User's Dataset**

# Define file path
file_path = "/mnt/data/cons22.txt"

# Load dataset into a DataFrame
df_user = pd.read_csv(file_path, sep="\s+", engine='python', 
                      names=['SIMTIME', 'source', 'Dest', 'needed', 'holding_time'])

# Convert necessary columns to numeric values
df_user['SIMTIME'] = df_user['SIMTIME'].astype(float)
df_user['source'] = df_user['source'].astype(int)
df_user['Dest'] = df_user['Dest'].astype(int)
df_user['needed'] = df_user['needed'].astype(int)
df_user['holding_time'] = df_user['holding_time'].astype(float)

# Display the first few rows
tools.display_dataframe_to_user(name="User Network Traffic Data", dataframe=df_user)

# Step 2: Train AI Model on User Dataset

# Define features and target variable
X_user = df_user[['SIMTIME', 'source', 'Dest', 'needed']]
y_user = df_user['holding_time']

# Split dataset into training and testing sets
X_train_user, X_test_user, y_train_user, y_test_user = train_test_split(X_user, y_user, test_size=0.2, random_state=42)

# Train AI model (Random Forest Regressor)
ai_model_user = RandomForestRegressor(n_estimators=100, random_state=42)
ai_model_user.fit(X_train_user, y_train_user)

# Predict and evaluate AI optimization
y_pred_user = ai_model_user.predict(X_test_user)
mae_user_optimized = mean_absolute_error(y_test_user, y_pred_user)

# Step 3: Visualize Before and After AI Optimization

plt.figure(figsize=(8, 5))
plt.bar(["Before AI Optimization", "After AI Optimization"], 
        [df_user['holding_time'].mean(), mae_user_optimized], 
        color=['red', 'green'])
plt.xlabel("Scenario")
plt.ylabel("Holding Time (seconds)")
plt.title("Comparison of Holding Time Before and After AI Optimization")
plt.show()

# Step 4: Feature Importance Analysis
feature_importances_user = ai_model_user.feature_importances_
feature_importance_dict_user = {
    'SIMTIME': feature_importances_user[0],
    'Source Node': feature_importances_user[1],
    'Destination Node': feature_importances_user[2],
    'Needed Slots': feature_importances_user[3],
}

plt.figure(figsize=(8, 5))
plt.bar(feature_importance_dict_user.keys(), feature_importance_dict_user.values(), color='blue')
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Predicting Holding Time")
plt.xticks(rotation=45)
plt.show()

# Returning the AI optimization results for comparison
mae_user_optimized, feature_importance_dict_user
