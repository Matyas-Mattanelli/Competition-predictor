import pandas as pd

from utils import export_dataset

# Load data set
df = pd.read_csv('data/filtered_cols_data_set.csv', sep=';', encoding='utf-8')

# Add average number of points and finals from previous competitions
df['Date'] = pd.to_datetime(df['Date']) # Convert to date
df.sort_values(['Taneční pár', 'Class', 'Type', 'Date'], inplace=True, ignore_index=True) # Sort values
df[['Obtained points prev', 'Obtained final prev']] = df.groupby(['Taneční pár', 'Class', 'Type'])[['Obtained points', 'Obtained final']].shift(1) # Helper column to calculate the expanding window
df['First comp in class'] = df['Obtained points prev'].isna().astype(int) # Indicator for a first competition in the class
df.loc[df['Obtained points prev'].isna(), ['Obtained points prev', 'Obtained final prev']] = 0 # Fill missing values with zero
df[['Average points', 'Average finals']] = df.groupby(['Taneční pár', 'Class', 'Type'])[['Obtained points prev', 'Obtained final prev']].expanding(1).mean().values

# Add number of competitions in a given class
df['Number of participations'] = df.groupby(['Taneční pár', 'Class', 'Type'])['Taneční pár'].expanding(1).count().values - 1

# Add number of days in the current class
df['Days in class'] = df['Date'] - df.groupby(['Taneční pár', 'Class', 'Type'])['Date'].transform('min')
df['Days in class'] = df['Days in class'].dt.days # Convert to integer

# Add number of days since first competition
df['Days since first comp'] = df['Date'] - df.groupby(['Taneční pár', 'Type'])['Date'].transform('min')
df['Days since first comp'] = df['Days since first comp'].dt.days # Convert to integer
 
# Add number of competitions in previous class
df['Previous class'] = df['Class'].map({'E':None, 'D':'E', 'C':'D', 'B':'C', 'A':'B'})
last_day_in_class = df.groupby(['Taneční pár', 'Class', 'Type'])['Date'].idxmax() # Find the row id of last day in each class for each pair and type
last_day_in_class_idx = last_day_in_class.reindex(pd.MultiIndex.from_frame(df.loc[:, ['Taneční pár', 'Previous class', 'Type']], names=['Taneční pár', 'Class', 'Type'])).values # Get the previous value for all rows
df[['Previous number of participations', 'Days in previous class']] = df[['Number of participations', 'Days in class']].reindex(pd.Index(last_day_in_class_idx), fill_value=0).values

# Adjust country
df['Country'] = df['Country'].replace({'SK':'SVK', 'PL':'POL', 'D':'GER', 'A':'AUT', 'H':'HUN'}) # Unify country names
df.loc[df['Country'].isin(['UKR', 'LTU', 'SLO', 'XXX','RUS', 'FRA', 'I', 'ROM', 'CRO', 'RO', 'CR', 'CK']), 'Country'] = 'Other' # Group countries with low occurences

# Adjust club
club_counts = df['Club'].value_counts(dropna=False) # Get number of occurences for each club
club_other = list(club_counts.index[club_counts < 100]) # Get clubs with less than 100 occurences
df.loc[df['Club'].isin(club_other), 'Club'] = 'Other' # Group clubs with low occurences

# Add average rank in the current class
df['Rank prev'] = df.groupby(['Taneční pár', 'Class', 'Type'])['Rank'].shift(1, fill_value=0) # Helper column to calculate the expanding window
df['Average rank in current class'] = df.groupby(['Taneční pár', 'Class', 'Type'])['Rank prev'].expanding(1).mean().values

# Make the statistics also for each person in the pair?

### Finalize the data set ###

# Drop unnecessary columns
for col in ['Obtained points prev', 'Obtained final prev', 'Previous class', 'Rank prev', 'Taneční pár', 'Obtained points', 'Points after', 'Finals after', 'Rank', 'Finalist', 'Obtained final', 'Age group', 'Class', 'Type', 'Date', 'Title']:
    df.drop(col, axis=1, inplace=True)

# Rename columns
df.rename(columns={'Obtained final (custom)':'Final'}, inplace=True)

# Put the dependent variable at the beginning
cols = list(df.columns)
cols.remove('Final')
df = df.loc[:, ['Final'] + cols]

# Drop duplicates
df.drop_duplicates(inplace=True)

# Drop erroneous values
df = df.loc[df['Points before'] >= 0, :]

# Drop competitions with less than two participants
df = df.loc[df['N'] > 2, :]

# Export the data set
export_dataset(df, 'final', 'csv', 'Yes')