**LSTM:**

```python
@op("LSTM", weights=True)
def lstm(x, *, input_size=1024, hidden_size=1024, dropout=0.0):
    lstm = torch.nn.LSTM(input_size, hidden_size, dropout=dropout, batch_first=True)
    if input_size == 1:
        return lambda x: lstm(x.unsqueeze(-1))[1][0].squeeze(0)
    return lambda x: lstm(x)[1][0].squeeze(0)

```
