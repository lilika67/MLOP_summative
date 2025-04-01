
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('cropyield.csv')

# Split into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save to data/train/ and data/test/
train_df.to_csv('data/train/cropyield_train.csv', index=False)
test_df.to_csv('data/test/cropyield_test.csv', index=False)
