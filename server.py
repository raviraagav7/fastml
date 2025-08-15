from mcp.server.fastmcp import FastMCP
import json
import requests
from typing import List

#server created
mcp = FastMCP("ChurnAndBurn")

#create the tool
@mcp.tool()
def PredictChurn(data: List[dict]) -> str:
    payload = data[0]
    response = requests.post("http://127.0.0.1:8000", 
                             headers={"Accept": "application/json", "Content-Type": "application/json"},
                             data=json.dumps(payload))
    return "Yes — they are likely to churn." if response.json()['prediction'] == 1 else "Unlikely to churn."


if __name__ == "__main__":
    mcp.run()
    