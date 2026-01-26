from new_data_loader import DataLoader

loader = DataLoader("D:/fpa_data")

trades = loader.load_aggtrades(
    symbol="BTCUSDT",
    start_date="2024-01-01",
    end_date="2024-01-01",
)

print(trades.columns)
print(trades.head(10))

# tick size proxy
trades["price_diff"] = trades["price"].diff().abs()
print(trades["price_diff"].dropna().head(10))