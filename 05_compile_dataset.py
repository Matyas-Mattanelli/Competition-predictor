import pickle
import pandas as pd
import os

import utils

# Specify options
KIND = 'csv'
OVERWRITE = 'Yes'

# Get all processed files
processed_files = os.listdir('data/processed data')
if 'Errors.txt' in processed_files: # Do not consider the log file
    processed_files.remove('Errors.txt')

# Concatenate data
df = pd.DataFrame([])
for file in processed_files: # Loop through all processed files
    # Load the file
    with open(f'data/processed data/{file}', 'rb') as handle:
        month_data = pickle.load(handle)

    # Append the file to the data
    df = pd.concat([df, pd.DataFrame(month_data)], ignore_index=True)

# Export the raw data set
utils.export_dataset(df, 'raw', kind=KIND, overwrite=OVERWRITE)

### Create and adjust features ###

# Extract country and club
df['Country'] = df['Klub(stát)'].str.extract(r'\((.*)\)') 
df['Club'] = df['Klub(stát)'].str.extract(r'(.*) \(')

# Extract points and finals
df['Obtained final'] = df['Body a finále získané v soutěži'].str.contains('F').astype(int) # Indicator whether the final (F) was awarded
df['Obtained points'] = df['Body a finále získané v soutěži'].str.replace('(F)', '').astype(int) # Get number of obtained points
df['Points after'] = df['Body a finále po soutěži'].str.replace(r'\(F[0-9]*\)', '', regex=True).astype(int) # Get number of points after competition
df['Finals after'] = df['Body a finále po soutěži'].str.extract(r'\(F([0-9]*)\)').astype(int) # Get the number of finals after the competition
df['Points before'] = df['Points after'] - df['Obtained points'] # Calculate the points prior to the competition
df['Finals before'] = df['Finals after'] - df['Obtained final'] # Calculate the finals prior to the competition

# Adjust the rankings
df['Rank'] = df['Umístění'].apply(utils.convert_rank) # Convert ranks to numbers

# Add an indicator whether the participant is a finalist or not
df['Finalist'] = ((df['Obtained final'] == 1) | df['Rank'].isin([1, 2, 3, 4])).astype(int) # All with an obtained final must be finalists. Also, there are at least 4 people in the final (less only when there is less than 4 total participants)
for idx in df.index:
    if df.loc[idx, 'Finalist'] != 1: # Skip already identified finalists
        if (df.loc[idx, 'Category'] == df.loc[idx - 1, 'Category']) and (df.loc[idx, 'Event'] == df.loc[idx - 1, 'Event']): # Proceed only if we are still in the same category
            if df.loc[idx - 1, 'Finalist'] == 1: # Can only be a finalist if the person above you is a finalist
                ties = 0 # Calculate number of ties
                while df.loc[idx, 'Rank'] == df.loc[idx + ties + 1, 'Rank']: # Keep increasing index until a non-tied participant is seen
                    ties += 1 
                if df.loc[idx - 1, 'Obtained points'] - df.loc[idx, 'Obtained points'] <= ties + 1: # Check if the person above differs in points only by number of defeated participants
                    df.loc[idx, 'Finalist'] = 1 # Indicate that the participant is a finalist

# Add a custom F check (mostly for class E)
df['Unique clubs'] = df.groupby(['Event', 'Category'])['Club'].transform('nunique')
df['Obtained final (custom)'] = df.apply(utils.is_final, axis=1)

# Split Category
df[['Age group', 'Class', 'Type']]= df['Category'].str.split('-', expand=True)

# Get date
df[['Date', 'Title']] = df['Event'].str.split(' - ', n=1, expand=True)
df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y') # Covert to date

# Export the full data set
utils.export_dataset(df, 'full', kind=KIND, overwrite=OVERWRITE)

# Drop unnecessary columns
for col in ['Klub(stát)', 'Body a finále získané v soutěži', 'Body a finále po soutěži', 'Umístění', 'Unique clubs', 'Event', 'Category', 'Judges']:
    df.drop(col, axis=1, inplace=True)

# Export the data set with filtered columns
utils.export_dataset(df, 'filtered_cols', kind=KIND, overwrite=OVERWRITE)
