from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import pandas as pd
import os


def load_csv_to_db(file_path):
    df = pd.read_csv(file_path)

    file_name = os.path.splitext(file_path.name)[0]
    if os.path.exists(f"files/{file_name}.db"):
        os.remove(f"files/{file_name}.db")
    engine = create_engine(f"sqlite:///files/{file_name}.db")
    df.to_sql(file_name, engine, index=False)
    db = SQLDatabase(engine=engine)
    print(db.get_usable_table_names())
    return db
