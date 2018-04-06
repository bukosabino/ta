import pandas as pd

# TODO: Stochastic oscillator / True strength index (TSI) / Ultimate oscillator / Williams %R (%R)

def rsi(close, n=14):
    """Relative Strength Index (RSI)
    """
    diff = close.diff()
    which_dn = diff < 0
    
    up, dn = diff, diff*0
    up[which_dn], dn[which_dn] = 0, -up[which_dn]
    
    emaup = up.ewm(n).mean()
    emadn = dn.ewm(n).mean()
    
    rsi = 100 * emaup/(emaup + emadn)
    return pd.Series(rsi, name='rsi')

    
def money_flow_index(high, low, close, volume, n=14):
    """Money Flow Index (MFI)
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi
    https://en.wikipedia.org/wiki/Money_flow_index
    """
    
    # 0 Prepare dataframe to work
    df = pd.DataFrame([high, low, close, volume]).T
    df.columns = ['High', 'Low', 'Close', 'Volume']
    df['Up_or_Down'] = 0
    df.loc[(df['Close'] > df['Close'].shift(1)), 'Up_or_Down'] = 1
    df.loc[(df['Close'] < df['Close'].shift(1)), 'Up_or_Down'] = 2

    # 1 typical price
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    
    # 2 money flow
    mf = tp * df['Volume']

    # 3 positive and negative money flow with n periods
    df['1p_Positive_Money_Flow'] = 0.0
    df.loc[df['Up_or_Down'] == 1, '1p_Positive_Money_Flow'] = mf
    n_positive_mf = df['1p_Positive_Money_Flow'].rolling(n).sum()

    df['1p_Negative_Money_Flow'] = 0.0
    df.loc[df['Up_or_Down'] == 2, '1p_Negative_Money_Flow'] = mf
    n_negative_mf = df['1p_Negative_Money_Flow'].rolling(n).sum()

    # 4 money flow index
    mr = n_positive_mf / n_negative_mf
    mr = (100 - (100 / (1 + mr)))
    return pd.Series(mr, name='mfi_'+str(n))

