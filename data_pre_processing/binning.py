import pandas as pd

def binning_bins(df, col, nr_bins):
  df['{}_binned_categories'.format(col)] = pd.cut(df[col], bins=nr_bins)
  df['{}_binned_labels'.format(col)] = pd.cut(df[col], bins=nr_bins, labels=False)

  return df

def binning_quantiles(df, col, nr_bins):
  df["{}_binned_categories_q".format(col)] = pd.qcut(df[col], q=nr_bins)
  df["{}_binned_labels_q".format(col)] = pd.qcut(df[col], q=nr_bins, labels=False)

  return df

def binning_ranges(df, col, ranges):
  ranges_tuples = []
  for r in ranges:
    ranges_tuples.append((r[0], r[1]))

  bins = pd.IntervalIndex.from_tuples(ranges_tuples)

  df['{}_binned_categories_q'.format(col)] = pd.cut(df[col], bins)

  x = pd.cut(df[col], bins)
  categories = []
  for i in len(set(df['{}_binned_labels_q'.format(col)])):
    categories.append(i)
  x.categories = categories
  
  df['{}_binned_labels_q'.format(col)] = x

  return df