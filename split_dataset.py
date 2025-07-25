import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV
df = pd.read_csv('brainage_classification.csv')

# First, split off the test set (30%)
train_val, test = train_test_split(
    df, test_size=0.3, stratify=df['Class'], random_state=42
)

# Now split train_val into train (60/90 = 66.67%) and val (10/90 = 11.11%)
train, val = train_test_split(
    train_val, test_size=1/6, stratify=train_val['Class'], random_state=42
)

# Save splits
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
val.to_csv('val.csv', index=False)

print(f"Train: {len(train)}, Test: {len(test)}, Val: {len(val)}") 