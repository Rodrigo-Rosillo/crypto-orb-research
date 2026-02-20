# Emergency: Flatten all futures positions

Use this standalone script to close all open Binance USD-M Futures positions at market with `reduceOnly=true`.

Testnet/demo:

```bash
BINANCE_TESTNET_API_KEY=... BINANCE_TESTNET_API_SECRET=... python scripts/emergency_flatten.py --testnet
```

Mainnet:

```bash
BINANCE_API_KEY=... BINANCE_API_SECRET=... python scripts/emergency_flatten.py
```

Docker (works even if the main trader container is dead):

Testnet/demo:

```bash
BINANCE_TESTNET_API_KEY=... BINANCE_TESTNET_API_SECRET=... docker compose run --rm trader python scripts/emergency_flatten.py --testnet
```

Mainnet:

```bash
BINANCE_API_KEY=... BINANCE_API_SECRET=... docker compose run --rm trader python scripts/emergency_flatten.py
```
