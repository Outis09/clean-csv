import sys
import os
from pathlib import Path
import pandas as pd
import time
import numpy as np
import logging

# exit program if no argument or more than one is given
if len(sys.argv) != 2:
    if len(sys.argv) < 2:
        print('You did not add a csv file when running this program.')
    else:
        print(f'You entered {len(sys.argv) - 1} arguments.')
    print('This program only accepts one csv file as argument as follows: python3 cleanCSV.py [path/to/your/data.csv]')
    sys.exit()

# get filepath
filepath = Path(sys.argv[1])

# convert all filepaths to absolute
abs_filepath = Path(filepath).resolve()

# set up logging
logger_name = str(abs_filepath)
logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)

# file for logging
log_folder = Path('./logs')
if log_folder.exists():
    log_filepath = log_folder.joinpath('cleanCSV.logs')
else:
    Path.mkdir('./logs')
    log_filepath = log_folder.joinpath('cleanCSV.logs')

# create handler to write logs to file
file_handler = logging.FileHandler(log_filepath, mode="a", encoding='utf-8')
# set up file formatter
file_Formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s")
file_handler.setFormatter(file_Formatter)

# create handler to write logs to console
console_handler = logging.StreamHandler()
# set up console formatter
console_formatter = logging.Formatter("%(message)s")
# add formatter to console handler
console_handler.setFormatter(console_formatter)

# add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# check if argument supplied is a csv
if filepath.suffix != '.csv':
    logger.error('The argument supplied is not a csv file.')
    sys.exit()

# load csv into pandas dataframe
try:
    print(f'Loading file from {abs_filepath} ...')
    df = pd.read_csv(abs_filepath)
    time.sleep(0.5)
    logger.info('Loaded file')
except Exception as e:
    logger.error(f'An error occured: {e}')
    sys.exit()

# check if csv file is empty (only headers)
if len(df) == 0:
    logger.error('The file has only columns but no rows.')
    sys.exit()

# copy data to be used for cleaning
cleanDF = df.copy()

# data profiling
print('\n')
print('Data Profile')

# get number of columns and rows
rows,cols = cleanDF.shape
logger.info(f'The data has {cols} columns and {rows} rows.')

# check for duplicates
duplicateCount = cleanDF.duplicated().sum()
logger.info(f'The data has {duplicateCount} duplicates.')

# check for null
# nullcleanDF = cleanDF.isnull().sum()#.reset_index()
# nullcleanDF = pd.DataFrame(nullcleanDF.reset_index()).rename(columns={'index':'columns', 0:'nulls'})
# onlyNulls = nullcleanDF[nullcleanDF['nulls'] > 0]
nullColumns = cleanDF.columns[cleanDF.isnull().any()].to_list()
if len(nullColumns) == 0:
    logger.info('There are no nulls.')
else:
    logger.warning(f'The following columns have null records: {nullColumns}')
    

# standardize column headers
def standardize_headers(headers):
    standardized_headers = {}
    for header in headers:
        standardized = header.strip().lower().replace(' ', '_')
        standardized_headers[header] = standardized
    return standardized_headers

print('\n')
print('Standardizing column headers...')
print('The original column headers are:')
print(list(cleanDF.columns))

logger.info('Standardized column headers')
new_headers = standardize_headers(cleanDF.columns)
print(list(new_headers.values()))

# rename column headers
cleanDF.rename(columns=new_headers, inplace=True)

# dropping duplicates, if they exist
if duplicateCount > 0:
    print(f'Removing {duplicateCount} duplicates')
    keeps = None
    duplResp = 0
    while True:
        if keeps in ['f', 'first', 'l', 'last']:
            break
        elif duplResp < 3  and keeps not in ['f', 'first', 'l', 'last']:
            keeps = input("For each duplicate, do you want to keep the (f)irst or the (last): ")
            keeps = keeps.lower()
            duplResp += 1
        elif duplResp >= 3:
            logger.error('Wrong response entered 3 times.')
            print('Please restart the program.')
            sys.exit()

    before_drop = len(cleanDF)
    if keeps in ['l', 'last']:
        cleanDF.drop_duplicates(keep='last', inplace=True)
    else:
        cleanDF.drop_duplicates(inplace=True)
    afterdrop = before_drop - len(cleanDF)
    logger.info(f'{afterdrop} duplicates have been dropped.')

# data types
shouldBeNumeric = []
for column in cleanDF.columns:
    if column not in cleanDF.select_dtypes(include=np.number).columns.to_list():
        try:
            cleanDF[column] = pd.to_numeric(cleanDF[column]) # cleanDF[column].astype('int')
            logger.info(f'{column} has been converted to an integer.')
        except ValueError:
            continue

def checkOutliers(cleanDF, column):
    firstQuartile, thirdQuartile =  np.percentile(cleanDF[column], [25,75])
    iqr = thirdQuartile - firstQuartile
    lowerBoundary = firstQuartile - (1.5 * iqr)
    upperBoundary = thirdQuartile + (1.5 * iqr)
    # check for outliers
    outliers = cleanDF[(cleanDF[column] < lowerBoundary) | (cleanDF[column] > upperBoundary)]
    outlierCount = len(outliers)
    if outlierCount > 0:
        return True
    else:
        return False
    
# dropping or imputing nulls
# get numeric columns
numericColumns = cleanDF.select_dtypes(include= ['int64', 'float64']).columns.to_list()

for column in numericColumns:
    # check for nulls
    nullCount = cleanDF[column].isnull().sum()
    cleanDFLen = len(cleanDF)
    if nullCount == 0:
        continue
    nullPercentage = round((nullCount/cleanDFLen) * 100,2)
    logger.info(f'There are {nullCount} in the {column} column.')
    print(f'And these make up {nullPercentage}% of the values in that column.')
    nullCmd = None
    nullCmdCount = 0
    while True:
        if nullCmd in ['d', 'i']:
            break
        if nullCmd is None and nullCmdCount < 3:
            nullCmd = input('Would you like to (i)mpute these values or (d)elete them: ')
            nullCmdCount += 1
        elif nullCmd.lower() not in ['d', 'i'] and nullCmdCount < 3:
            nullCmd = input("You entered the wrong command. Enter 'i' to impute nulls and 'd' to delete nulls: ")
            nullCmdCount += 1
        else:
            logger.error("Entered wrong command three times.")
            print('Please restart the program')
            sys.exit()
    # drop nulls
    if nullCmd.lower() == 'd':
        cleanDF = cleanDF.dropna(subset=[column]).reset_index()
        logger.info(f'{nullCount} nulls dropped in {column} column.')
    else:
        time.sleep(0.5)
        print('Checking for outliers...')
        time.sleep(0.5)
        outlierStatus = checkOutliers(cleanDF,column)
        if outlierStatus:
            imputer = round(np.median(cleanDF[column]), 2)
            logger.info(f'Since there are outliers in the data, nulls will be imputed with the median.')
        else:
            imputer = round(np.mean(cleanDF[column]), 2)
            logger.info('There are no outliers in the data, so nulls will be imputed with the mean')
        cleanDF[column] = cleanDF[column].fillna(imputer)


# save cleaned file to csv in same directory as data
saveDir = abs_filepath.parent
fileName = abs_filepath.stem
saveName = saveDir.joinpath(f'{fileName}-clean.csv')
cleanDF.to_csv(saveName, index=False)
logger.info(f'Cleaned csv saved at {saveName}')