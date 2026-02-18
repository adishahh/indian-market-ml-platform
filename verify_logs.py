from config.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    print("Checking prediction logs...")
    result = conn.execute(
        text("SELECT * FROM prediction_logs ORDER BY id DESC LIMIT 5")
    )
    rows = result.fetchall()
    if not rows:
        print("No logs found.")
    else:
        for row in rows:
            print(f"ID: {row[0]}, Date: {row[2]}, Prediction: {row[3]}, Prob: {row[4]}")
