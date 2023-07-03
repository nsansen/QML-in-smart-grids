import numpy as np
import pandas as pd
def lstm_data(ms,df,splitsequence):
    df.index=pd.to_datetime(df["Unnamed: 0"], format = "%Y-%m-%d %H:%M:%S") 


    temp_df = pd.DataFrame({'Target':df[ms]})
    temp_df['Seconds'] = temp_df.index.map(pd.Timestamp.timestamp)
    temp_df["Seconds"]=temp_df["Seconds"]/temp_df['Seconds'].max()


    temp_df['Seconds sin'] = np.sin(temp_df['Seconds']) 
    temp_df['Seconds cos'] = np.cos(temp_df['Seconds']) 

    day = 60*60*24
    year = 365.2425*day
    temp_df['Day sin'] = np.sin(temp_df['Seconds'] * (2* np.pi / day))
    temp_df['Day cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / day))
    temp_df['Year sin'] = np.sin(temp_df['Seconds'] * (2 * np.pi / year))
    temp_df['Year cos'] = np.cos(temp_df['Seconds'] * (2 * np.pi / year))
    
    def df_to_X_y2(df, window_size=splitsequence):
        df_as_np = df.to_numpy()
        X = []
        y = []
        for i in range(len(df_as_np)-window_size):
            row = [r for r in df_as_np[i:i+window_size]]
            X.append(row)
            label = df_as_np[i+window_size][0]
            y.append(label)
        return np.array(X), np.array(y)
    x0, y0 = df_to_X_y2(temp_df)
    
    
    return x0, y0