import pandas as pd
import calendar, datetime

MONTH_DICT = dict((v,k) for k,v in enumerate(calendar.month_name))
FEATURE_COLUMNS = ['lead_avg_rank', 'lead_avg_points', 'lead_count',
                   'boulder_avg_rank', 'boulder_count',
                   'speed_avg_rank', 'speed_count']
GENDER_DF = pd.read_csv('ifsc_climbing_data/genders.csv')

def rename_columns(df):
    rename_dict = {name : name.lower().replace(' ', '_') for name in df.columns}
    return df.rename(columns=rename_dict)

def remove_youth(df):
    return df[~df.competition_title.str.lower().str.contains('youth')]


def get_end_day(string):
    value_list = re.split(' ', string)
    
    if len(value_list) == 6:
        end_day = int(value_list[3])
    else:
        end_day = None
        
    return end_day

def date_create(df):
    
    df['year'] = df.competition_date.str.slice(start = -4).astype('int')
    df['month_string'] = df.competition_date.str.extract('([A-za-z]+)')
    df['month'] = df.month_string.map(MONTH_DICT)
    df['day'] = df.competition_date.str.extract('(\d+)').astype('int')
    df['start_date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    return df

def date_filter(df, date):
    return df[df.start_date < date]

def data_cleaning(df, filter_dates = True, date = None):
    df_renamed = rename_columns(df)
    df_dedup = df_renamed.drop_duplicates()
    df_adult = remove_youth(df_dedup)
    df_date = date_create(df_adult)
    
    if filter_dates:
        date_filter_df = date_filter(df_date, date)
        return date_filter_df
    else:
        return df_date

def agg_join_data(lr_df, br_df, sr_df, predicting_comp):
    join_col = ['first', 'last']
    
    ld_agg_mean = lr_df[['first', 'last', 'rank', 'points']].groupby(join_col).mean().reset_index()
    ld_agg_count = lr_df[['first', 'last', 'rank']].groupby(join_col).count().reset_index()
    ld_agg_df = ld_agg_mean.merge(ld_agg_count, on = join_col).rename(columns = {'rank_x': 'lead_avg_rank',
                                                                           'points': 'lead_avg_points',
                                                                           'rank_y': 'lead_count'
                                                                          })
    br_agg_mean = br_df[['first', 'last', 'rank']].groupby(join_col).mean().reset_index()
    br_agg_count = br_df[['first', 'last', 'rank']].groupby(join_col).count().reset_index()
    br_agg_df = br_agg_mean.merge(br_agg_count, on = join_col).rename(columns = {'rank_x': 'boulder_avg_rank',
                                                                           'rank_y': 'boulder_count'
                                                                          })
    sr_agg_mean = sr_df[['first', 'last', 'rank']].groupby(join_col).mean().reset_index()
    sr_agg_count = sr_df[['first', 'last', 'rank']].groupby(join_col).count().reset_index()
    sr_agg_df = sr_agg_mean.merge(sr_agg_count, on = join_col).rename(columns = {'rank_x': 'speed_avg_rank',
                                                                           'rank_y': 'speed_count'
                                                                          })
    predicting_comp['full_name'] = predicting_comp['last'] + ' '  + predicting_comp['first']

    pred_aggs_raw = predicting_comp.merge(ld_agg_df, how = 'left', on = join_col
              ).merge(br_agg_df, how = 'left', on = join_col
              ).merge(sr_agg_df, how = 'left', on = join_col
              ).merge(GENDER_DF, on = ['full_name']
              )[['first', 'last', 'nation', 'rank', 'gender'] + FEATURE_COLUMNS]
    
    return pred_aggs_raw

def create_fill_value(column, value):
    if 'count' in column:
        return 0
    else:
        return value

def fill_features(df):
    max_values = df[FEATURE_COLUMNS].max()

    fill_dict = {column : create_fill_value(column, value) for 
                 column, value in zip(max_values.keys(), max_values)}

    return df.fillna(value = fill_dict)

def process_data(br_raw, lr_raw, sr_raw, cr_raw, date, comp_name):
    br_df = data_cleaning(br_raw, date = date)
    lr_df = data_cleaning(lr_raw, date = date)
    sr_df = data_cleaning(sr_raw, date = date)


    cr_df = data_cleaning(cr_raw, False)
    predicting_comp = cr_df[cr_df.competition_title == comp_name]

    pred_aggs_raw = agg_join_data(lr_df, br_df, sr_df, predicting_comp)

    pred_aggs = fill_features(pred_aggs_raw)

    pred_aggs['avg_rank_multi'] = pred_aggs.lead_avg_rank * pred_aggs.boulder_avg_rank * pred_aggs.speed_avg_rank

    return pred_aggs

