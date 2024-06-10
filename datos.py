import yfinance as yf
import pandas as pd

def baja_dato(ticker: str, start: str = "2000-01-01", end: str = None, interval: str = "1d") -> pd.Series:
    """
    Descarga los datos ajustados de cierre para un ticker específico.
    
    :param ticker: Símbolo del ticker.
    :param start: Fecha de inicio (formato 'AAAA-MM-DD'). Por defecto es '2000-01-01'.
    :param end: Fecha de fin (formato 'AAAA-MM-DD'). Por defecto es None (hasta la fecha actual).
    :param interval: Intervalo de los datos (por defecto '1d' para diario).
    :return: Serie de datos ajustados de cierre renombrada con el ticker.
    """
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval)
        return data["Adj Close"].rename(ticker)
    except Exception as e:
        print(f"Error descargando datos para {ticker}: {e}")
        return pd.Series(dtype='float64')

def baja_datos(tickers: list[str], start: str = "2000-01-01", end: str = None, interval: str = "1d") -> pd.DataFrame:
    """
    Descarga los datos ajustados de cierre para una lista de tickers.
    
    :param tickers: Lista de símbolos de tickers.
    :param start: Fecha de inicio (formato 'AAAA-MM-DD'). Por defecto es '2000-01-01'.
    :param end: Fecha de fin (formato 'AAAA-MM-DD'). Por defecto es None (hasta la fecha actual).
    :param interval: Intervalo de los datos (por defecto '1d' para diario).
    :return: DataFrame con los datos ajustados de cierre.
    """
    if not tickers:
        print("La lista de tickers está vacía.")
        return pd.DataFrame()

    try:
        data = yf.download(tickers, start=start, end=end, interval=interval)["Adj Close"]
        return data.dropna(axis=0)
    except Exception as e:
        print(f"Error descargando datos para los tickers: {e}")
        return pd.DataFrame()

tickers = ["MSFT", "GOOG", "EWZ", "BRK-B", "COST", "JNJ", "ZF=F", "ECH"]

if __name__ == "__main__":
    datos = baja_datos(tickers)
    print(datos)