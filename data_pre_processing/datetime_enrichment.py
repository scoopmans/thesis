import pandas as pd
import numpy as np
from datetime import date, datetime
from utils.report import type_cols

def make_date_hour(df, date_col=None):
  '''From a datetime variable it constructs an additional variable
  containing the hour of the day.

  Parameters:
  -----------
  df: pd.DataFrame
  date_col = str, containing the datetime column name
  '''
  if not date_col:
    cat_col, date_col, num_col = type_cols(df_test)
    date_col=date_col[0]

  df['hour'] = df[date_col].apply(lambda x : x.hour)
  return df

def cyclical_time(df, time_col):
  df['hr_sin'] = np.sin(df[time_col]*(2*np.pi/24))
  df['hr_cos'] = np.cos(df[time_col]*(2*np.pi/24))
  return df
  
def make_date_weekday(df, date_col=None, temp=False):
  '''From a datetime variable it constructs an additional variable
  containing a number for the day of the week (monday = 0,
  sunday = 6)

  Parameters:
  -----------
  df: pd.DataFrame
  date_col = str, containing the datetime column name
  '''
  if not date_col:
    cat_col, date_col, num_col = type_cols(df_test)
    date_col = date_col[0]
  
  if temp:
    df['weekday_temp'] = df[date_col].apply(lambda x : x.weekday())
  else:
    df['weekday'] = df[date_col].apply(lambda x : x.weekday())

  return df

def cyclical_weekday(df, day_col):
  df['weekday_sin'] = np.sin(df[day_col]*(2*np.pi/7))
  df['weekday_cos'] = np.cos(df[day_col]*(2*np.pi/7))
  return df

def make_date_day_of_month(df, date_col=None):
  '''From a datetime variable it constructs an additional variable
  containing the number for the day of the month.

  Parameters:
  -----------
  df: pd.DataFrame
  date_col = list of str, containing the datetime column name
  '''
  if not date_col:
    cat_col, date_col, num_col = type_cols(df_test)
    date_col = date_col[0]
  
  df['day'] = df[date_col].apply(lambda x : x.day)

  return df

def cyclical_monthday(df, day_col):
  df['day_sin'] = np.sin(df[day_col]*(2*np.pi/30))
  df['day_cos'] = np.cos(df[day_col]*(2*np.pi/30))
  return df

def make_date_week_number(df, date_col=None):
  '''From a datetime variable it constructs an additional variable
  containing the number for the week of the year.

  Parameters:
  -----------
  df: pd.DataFrame
  date_col = list of str, containing the datetime column name
  '''
  if not date_col:
    cat_col, date_col, num_col = type_cols(df_test)
    date_col = date_col[0]
  
  df['week_number'] = df[date_col].apply(lambda x : x.isocalendar()[1])
  return df

def cyclical_week_number(df, week_nr):
  df['weeknr_sin'] = np.sin((df[week_nr]-1)*(2*np.pi/52))
  df['weeknr_cos'] = np.cos((df[week_nr]-1)*(2*np.pi/52))
  return df

def make_date_month(df, date_col=None):
  '''From a datetime variable it constructs an additional variable
  containing the number for the month of the year.

  Parameters:
  -----------
  df: pd.DataFrame
  date_col = str, containing the datetime column name
  '''
  if not date_col:
    cat_col, date_col, num_col = type_cols(df_test)
    date_col = date_col[0]
  
  df['month'] = df[date_col].apply(lambda x : x.month)
  return df

def cyclical_month(df, month_col):
  df['month_sin'] = np.sin((df[month_col]-1)*(2*np.pi/12))
  df['month_cos'] = np.cos((df[month_col]-1)*(2*np.pi/12))
  return df

def make_date_year(df, date_col=None):
  '''From a datetime variable it constructs an additional variable
  containing the year number.

  Parameters:
  -----------
  df: pd.DataFrame
  date_col = str, containing the datetime column
  '''
  if not date_col:
    cat_col, date_col, num_col = type_cols(df_test)
    date_col = date_col[0]
  
  df['year'] = df[date_col].apply(lambda x : x.year)
  return df

def get_season(datum, country=None):
  Y=2000

  if isinstance(datum, datetime):
    datum = datum.date()
  datum = datum.replace(year=Y)
  
  if date(Y,  1,  1) <= datum <= date(Y,  3, 20):
    return 'winter'
  elif date(Y,  3, 21) <= datum <= date(Y,  6, 20):
    return 'spring'
  elif date(Y,  6, 21) <= datum <= date(Y,  9, 22):
    return 'summer'
  elif date(Y,  9, 23) <= datum <= date(Y, 12, 20):
    return 'autumn'
  elif date(Y, 12, 21) <= datum <= date(Y, 12, 31):
    return 'winter'
  else:
    return np.nan

def season_to_num(season):
  if season == 'winter':
    return 0
  elif season == 'autumn':
    return 3
  elif season == 'spring':
    return 1
  elif season == 'summer':
    return 2

def make_date_season(df, date_col=None, country_col=None):
  '''From a datetime variable it constructs an additional variable
  containing the season number where:
    winter = 0
    spring = 1
    summer = 2
    autumn = 3
    
  Parameters:
  -----------
  df: pd.DataFrame
  date_col = str, containing the datetime column name
  '''

  if not date_col:
    cat_col, date_col, num_col = type_cols(df_test)
    date_col = date_col[0]

  if country_col:
    df['season'] = df.apply(lambda x : get_season(x[date_col], x[country_col]), axis=1)
  else:
    df['season'] = df[date_col].apply(lambda x : get_season(x))

  df['season'] = df['season'].apply(lambda x: season_to_num(x))
  return df

def cyclical_season(df, season_col):
  df['season_sin'] = np.sin(df[season_col]*(2*np.pi/4))
  df['season_cos'] = np.cos(df[season_col]*(2*np.pi/4))
  return df

def get_quarter(datum):
  '''From a datetime entry it returns a number for which quarter it belongs to.
    
  Parameters:
  -----------
  df: pd.DataFrame
  '''
  Y=2000
  if isinstance(datum, datetime):
    datum = datum.date()

  datum = datum.replace(year=Y)

  if date(Y,  1,  1) <= datum <= date(Y,  3, 31):
    return 1
  elif date(Y,  4, 1) <= datum <= date(Y,  6, 30):
    return 2
  elif date(Y,  7, 1) <= datum <= date(Y,  9, 30):
    return 3
  elif date(Y,  10, 1) <= datum <= date(Y, 12, 31):
    return 4
  else:
    return np.nan
  
def make_date_quarter(df, date_col=None):
  '''From a datetime variable it constructs an additional variable
  containing the quarter of the year.
    
  Parameters:
  -----------
  df: pd.DataFrame
  date_col = str, containing the datetime column name
  '''
  if not date_col:
    cat_col, date_col, num_col = type_cols(df_test)
    date_col = date_col[0]


  df['quarter'] = df[date_col].apply(lambda x : get_quarter(x))

  return df

def cyclical_quarter(df, quarter_col):
  df['quarter_sin'] = np.sin((df[quarter_col]-1)*(2*np.pi/4))
  df['quarter_cos'] = np.cos((df[quarter_col]-1)*(2*np.pi/4))
  return df

def weekend(weekday):
  if weekday > 4:
    return 1
  else:
    return 0

def weekend_or_not(df, date_col=None):
  '''From a datetime variable it constructs an additional variable
  indicating whether it is the weekend or not.
    
  Parameters:
  -----------
  df: pd.DataFrame
  date_col = str, containing the datetime column name
  '''
  if not date_col:
    cat_col, date_col, num_col = type_cols(df_test)
    date_col = date_col[0]
  
  df = make_date_weekday(df, date_col, temp=True)
  
  df['weekend'] = df['weekday_temp'].apply(lambda x : weekend(x))
  
  return df.drop(['weekday_temp'], axis=1)

def columns_to_datetime(df, year=None, month=None, day=None):
  if not year:
    for encoding in ['year', 'Year', 'YEAR']:
      if encoding in df.columns:
        year = encoding
          
  if not month:
    for encoding in ['month', 'Month', 'MONTH']:
      if encoding in df.columns:
        month = encoding
          
  if not day:
    for encoding in ['day', 'Day', 'DAY']:
      if encoding in df.columns:
        day = encoding
  
  date = pd.to_datetime(df[[year, month, day]],format='%Y%m%d',errors="coerce")
  df['Date'] = date
  return df
