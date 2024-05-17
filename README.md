# LynxKite 2024

This is an experimental rewrite of [LynxKite](https://github.com/lynxkite/lynxkite).
It is not compatible with the original LynxKite. For the design goals see
[LynxKite 2024 Roadmap](https://docs.google.com/document/d/12uhjib6M0bgdA9Ch5h4X8eBinmYOnzRa8bPwDWY0jtg/edit).

## Installation

To run the backend:

```bash
pip install -r requirements.txt
uvicorn server.main:app --reload
```

To run the frontend:

```bash
cd web
npm i
npm run dev
```
