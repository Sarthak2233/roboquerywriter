from sqlalchemy import create_engine, inspect, MetaData

# Replace with your actual database URL
DATABASE_URL = "mysql+mysqlconnector://root:admin12345@localhost:3306/mywebapp"

# Create an engine
engine = create_engine(DATABASE_URL)
inspector =   inspect(engine)
metadata = MetaData()

# Reflect the schema
metadata.reflect(bind=engine)
tables = inspector.get_table_names()

all_datas = ""
print(tables)


for table in tables:
    print(f"\nTable: {table}")
    columns = inspector.get_columns(table)
    for column in columns:
        print(f" - {column['name']}: {column['type']}")


# Test connection
relationships = []

for table in tables:
    foreign_keys = inspector.get_foreign_keys(table)
    if foreign_keys:
        print(f"\nForeign keys in table '{table}':")
        for fk in foreign_keys:
            print(f" - Column: {fk['constrained_columns']} references {fk['referred_table']}({fk['referred_columns']})")
            relationships.append({
                "table": table,
                "column": fk["constrained_columns"],
                "referred_table": fk["referred_table"],
                "referred_column": fk["referred_columns"]
            })


# Summary of relationships
print("\nRelationships:")
for relationship in relationships:
    print(f"{relationship['table']}({', '.join(relationship['column'])}) -> "
          f"{relationship['referred_table']}({', '.join(relationship['referred_column'])})")
try:
    connection = engine.connect()
    print("Connection successful!")
    connection.close()
except Exception as e:
    print(f"Error: {e}")
