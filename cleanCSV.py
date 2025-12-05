import sys
from pathlib import Path
import pandas as pd
import time
import numpy as np
import logging
import logging.config
import json

# check arguments supplied
if len(sys.argv) != 2:
    if len(sys.argv) < 2:
        print('You did not add a csv file when running this program.')
    else:
        print(f'You entered {len(sys.argv) - 1} arguments.')
    print('This program only accepts one csv file as argument as follows: python3 cleanCSV.py [path/to/your/data.csv]')
    sys.exit(1)

# get filepath
filepath = Path(sys.argv[1])

# convert all filepaths to absolute
abs_filepath = Path(filepath).resolve()
# setup module-level logger to handle logs and propagate to root logger
logger = logging.getLogger(__name__)
# adapt logger to add name of csv file to log message
logger = logging.LoggerAdapter(logger, {'filepath': str(abs_filepath)})

# set up root logger
def setup_logging():
    # folder for logging
    log_folder = Path('./logs')
    if not log_folder.exists():
        log_folder.mkdir(parents=True, exist_ok=True)

    configFile = Path('logConfig.json')
    with open(configFile, "r") as file:
        config = json.load(file)
    logging.config.dictConfig(config)

    logger = logging.getLogger()

def profile_data(df: pd.DataFrame):
    print('Data Profile')

    # get number of columns and rows
    rows,cols = df.shape
    logger.info(f'The data has {cols} columns and {rows} rows.')

    # check for duplicates
    duplicateCount = df.duplicated().sum()
    logger.info(f'The data has {duplicateCount} duplicates.')

    # check for null
    nullColumns = df.columns[df.isnull().any()].to_list()
    if len(nullColumns) == 0:
        logger.info('There are no nulls.')
    else:
        logger.warning(f'The following columns have null records: {nullColumns}')

    
# standardize column headers
def standardize_headers(headers: list) -> list:
    standardized_headers = {}
    for header in headers:
        standardized = header.strip().lower().replace(' ', '_')
        standardized_headers[header] = standardized
    return standardized_headers

def take_input(expected: list,tries: int, input_string) -> tuple[str, None] | tuple[None, int]:
    keeps = None
    respCount = 0
    while True:
        if keeps in expected:
            logger.info('Some test')
            return keeps, None
        elif respCount < tries and keeps not in expected:
            keeps = input(input_string)
            keeps = keeps.lower()
            respCount += 1
        elif respCount >= tries:
            return None, respCount
        
def drop_duplicates(df: pd.DataFrame, keep_inst: str) -> tuple[pd.DataFrame, str]:
    # length of df before dropping duplicates
    before_dupl_drop = len(df)
    # drop duplcates
    df = df.drop_duplicates(keep=keep_inst)
    # length of df after dropping duplicates
    after_dupl_drop = len(df)
    dupl_removed = before_dupl_drop - after_dupl_drop
    return df, dupl_removed


def checkOutliers(cleanDF: pd.DataFrame, column: str) -> bool:
    colSeries = cleanDF[column].dropna()
    firstQuartile, thirdQuartile =  np.percentile(colSeries, [25,75])
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


def handle_nulls(instruction: str, df: pd.DataFrame, column: str) -> pd.DataFrame:
    if instruction == 'drop':
        df[column] = df.dropna(subset=[column]).reset_index()
        logger.info(f"Dropped nulls in '{column}' column.")
        return df
    else:
        time.sleep(0.5)
        print(f"Checking for outliers in '{column}'...")
        time.sleep(0.5)
        outlier_status = checkOutliers(df, column)
        if outlier_status:
            imputer = round(np.median(df[column]), 2)
            logger.info(f'Since there are outliers in the data, nulls will be imputed with the median.')
        else:
            imputer = round(np.mean(df[column]))
            logger.info('There are no outliers in the data, so nulls will be imputed with the mean')
        df[column] = df[column].fillna(imputer)
        logger.info(f"Imputed nulls in the '{column}' column")
        return df


def save_cleaned_csv(df: pd.DataFrame, filepath: Path):
    # save cleaned file to csv in same directory as data
    save_dir = filepath.parent
    file_name = filepath.stem
    save_name = save_dir.joinpath(f'{file_name}-clean.csv')
    df.to_csv(save_name, index=False)
    logger.info(f'Cleaned csv saved at {save_name}')


def main():
    # setup root logger
    setup_logging()

    logger.info(f'Initialized logging for cleaning {abs_filepath}')

    # load csv
    try:
        df = pd.read_csv(abs_filepath)
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
    profile_data(cleanDF)

    # drop columns with only nulls
    nullCols = [column for column in cleanDF.columns if cleanDF[column].isnull().all()]
    if nullCols:
        logger.info(f'The following columns contain only nulls and will be dropped: {nullCols}')
        cleanDF = cleanDF.drop(columns=nullCols)

    print('\n')
    print('Standardizing column headers...')
    print('The original column headers are:')
    print(list(cleanDF.columns))

    logger.info('Standardized column headers')
    new_headers = standardize_headers(cleanDF.columns)
    print(list(new_headers.values()))

    # rename column headers
    cleanDF.rename(columns=new_headers, inplace=True)

    # handle duplicates
    duplicateCount = df.duplicated().sum()
    if duplicateCount > 0:
        expected = ['first', 'last']
        tries = 3
        input_string = "For each duplicate, would you like to keep the 'first' or 'last': "
        handle_duplicates, input_attempts = take_input(expected, tries, input_string)
        if input_attempts:
            logger.error(f'Wrong input entered {tries} times for handling duplicates.')
            print('Please restart the program')
            sys.exit()
        logger.info(f'Dropping duplicates, and keeping the {handle_duplicates}...')
        cleanDF, dupl_removed = drop_duplicates(cleanDF, handle_duplicates)
        logger.info(f'Kept {handle_duplicates} instances, and removed {dupl_removed} duplicates.')

    # dropping/imputing nulls in numeric columns
    numericColumns = cleanDF.select_dtypes(include= ['int64', 'float64']).columns.to_list()

    for column in numericColumns:
        # check for nulls
        nullCount = cleanDF[column].isnull().sum()
        cleanDFLen = len(cleanDF)
        if nullCount == 0:
            continue
        nullPercentage = round((nullCount/cleanDFLen) * 100,2)
        logger.info(f'There are {nullCount} nulls in the {column} column.')
        print(f'And these make up {nullPercentage}% of the values in that column.')

        expected = ['impute', 'keep', 'drop']
        tries = 3
        input_string = f"Would you like to 'impute', 'keep', or 'drop' nulls in the '{column}' column: "
        null_cmd, input_attempts = take_input(expected, tries, input_string)

        if input_attempts:
            logger.error('Wrong input entered {tries} for handling nulls.')
            print('Please restart the program')

        if null_cmd == 'keep':
            logger.info(f"Keeping nulls in '{column}' column.")
            continue
        else:
            cleanDF = handle_nulls(null_cmd, cleanDF, column)

    # save cleaned file to csv
    save_cleaned_csv(cleanDF, abs_filepath)


if __name__ == "__main__":
    main()