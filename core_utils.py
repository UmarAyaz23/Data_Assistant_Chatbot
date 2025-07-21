from langchain.schema import Document

import pandas as pd
import numpy as np
import os
import re


def preprocessing(raw_data, cleaned_data):
    try:
        if not os.path.exists(cleaned_data):
            print("Engineered file not found. Starting preprocessing...")

            df = pd.read_csv(raw_data)
            #df_sampled = df.sample(frac=0.25, random_state=42)  # 25% sample
            #df_sampled.to_csv("Orders_0.25.csv", index=False)

            df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], format='%m/%d/%Y', errors='coerce')
            df['Creation Date'] = pd.to_datetime(df['Creation Date'], format='%m/%d/%Y', errors='coerce')

            # Add Purchase Month, Quarter, and Year
            df['Purchase Month'] = df['Purchase Date'].dt.month
            df['Purchase Quarter'] = df['Purchase Date'].dt.to_period('Q').astype(str)
            df['Purchase Year'] = df['Purchase Date'].dt.year

            # Handle currency
            currency_cols = ['Unit Price', 'Total Price']
            for col in currency_cols:
                if col in df.columns:
                    df[col] = df[col].replace(r'[\$,]', '', regex=True).astype(float)

            # Handle categorical data
            categorical_cols = [
                'Fiscal Year', 'LPA Number', 'Purchase Order Number', 'Requisition Number',
                'Acquisition Type', 'Sub-Acquisition Type', 'Acquisition Method', 'Sub-Acquisition Method',
                'Department Name', 'Supplier Name', 'Supplier Qualifications', 'Classification Codes',
                'Normalized UNSPSC', 'Commodity Title', 'Class', 'Class Title',
                'Family', 'Family Title', 'Segment', 'Segment Title'
            ]
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')

            # Handle zip code
            df['Supplier Zip Code'] = df['Supplier Zip Code'].astype(str).str.zfill(5)

            # Extract latitude/longitude from Location field
            def extract_lat_long(location_str):
                if pd.isna(location_str):
                    return (np.nan, np.nan)
                match = re.search(r'\(([^,]+), ([^,]+)\)', location_str)
                return (float(match.group(1)), float(match.group(2))) if match else (np.nan, np.nan)

            if 'Location' in df.columns:
                df[['Latitude', 'Longitude']] = df['Location'].apply(extract_lat_long).apply(pd.Series)

            # Handle CalCard column
            df['CalCard'] = df['CalCard'].fillna('NO').map({'NO': False, 'YES': True})

            # Quantity to float
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')

            # Fill NA in some categorical fields
            categorical_fill_cols = [
                'Supplier Qualifications', 'Sub-Acquisition Type',
                'Sub-Acquisition Method', 'Classification Codes', 'Commodity Title'
            ]
            for col in categorical_fill_cols:
                if col in df.columns:
                    if isinstance(df[col].dtype, pd.CategoricalDtype):
                        if "Unknown" not in df[col].cat.categories:
                            df[col] = df[col].cat.add_categories("Unknown")
                    df[col] = df[col].fillna("Unknown")

            # Save the cleaned data
            df.to_parquet(cleaned_data, index=False)
            print(f'Preprocessing Complete, Data saved to: {cleaned_data}')
            return df

        else:
            print(f'Engineered Data already exists: {cleaned_data}. Skipping Preprocessing')
            df = pd.read_parquet(cleaned_data)
            return df

    except Exception as e:
        print(f'An Error occurred during preprocessing: {e}')
        return None

RAW_CSV = "Orders.csv"
CLEANED_PARQUET = "Orders_Cleaned.parquet"
df = preprocessing(RAW_CSV, CLEANED_PARQUET)
print(df.shape[0])


#--------------------------------------------------------------------------EMBEDDINGS--------------------------------------------------------------------------
def row_to_document(row):
    # Explicitly convert non-string or NaN descriptions to a default
    item_description = row.get('Item Description')
    if pd.isna(item_description) or not isinstance(item_description, str):
        page_content = "No description available"
    else:
        page_content = item_description.strip()

    # Extract metadata safely and consistently
    metadata = {
        "Purchase Date": row['Purchase Date'].strftime('%Y-%m-%d') if pd.notna(row['Purchase Date']) else "Unknown",
        "Creation Date": row['Creation Date'].strftime('%Y-%m-%d') if pd.notna(row['Creation Date']) else "Unknown",
        "Total Price": float(row['Total Price']) if pd.notna(row['Total Price']) else 0.0,
        "Item Name": row.get('Item Name', 'Unknown'),
        "Department Name": row.get('Department Name', 'Unknown'),
        "Fiscal Year": row.get('Fiscal Year', 'Unknown'),
        "Acquisition Type": row.get('Acquisition Type', 'Unknown'),
        "Supplier Name": row.get('Supplier Name', 'Unknown'),
        "Supplier Zip Code": row.get('Supplier Zip Code', 'Unknown'),
        "CalCard": bool(row['CalCard']) if pd.notna(row.get('CalCard')) else False,
        "Quantity": float(row['Quantity']) if pd.notna(row['Quantity']) else 0,
        "Segment Title": row.get('Segment Title', 'Unknown'),
        "Family Title": row.get('Family Title', 'Unknown'),
        "Class Title": row.get('Class Title', 'Unknown'),
        "Commodity Title": row.get('Commodity Title', 'Unknown'),
        "Purchase Month": row.get('Purchase Month', 'Unknown'),
        "Purchase Quarter": row.get('Purchase Quarter', 'Unknown'),
        "Purchase Year": row.get('Purchase Year', 'Unknown'),
        "Latitude": row.get('Latitude', None),
        "Longitude": row.get('Longitude', None)
    }

    return Document(page_content = page_content, metadata = metadata)