import os
import json
import boto3
import string
import logging
import time
import io
import csv
import uuid 

logger = logging.getLogger(name='genbisql-dev')

bedrock = boto3.client("bedrock-runtime")
redshift_serverless = boto3.client("redshift-serverless")

#---- Save content to S3 ----
def save_to_s3(content: str, prefix: str = "llm-output", expires_in: int = 60000) -> str:
    """
    Save content to S3 and return a pre-signed URL (default expiration: 1 hour)
    """
    s3 = boto3.client("s3")
    bucket = os.environ.get("OUTPUT_BUCKET")
    key = f"{prefix}/{uuid.uuid4()}.txt"

    # Upload the object
    s3.put_object(Bucket=bucket, Key=key, Body=content.encode("utf-8"))

    # Generate a pre-signed URL
    presigned_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in  # in seconds
    )

    return presigned_url

#---- Estimate tokens based on text length----
def estimate_tokens(text):
    """Rough estimate: 1 token ~ 4 characters"""
    return len(text) // 4

# ---- Truncate CSV Lines ----
def truncate_csv_lines(csv_string, max_token_limit):
    """
    Truncate CSV string to fit within the max token limit
    """
    lines = csv_string.strip().split('\n')
    header = lines[0]
    data_lines = lines[1:]

    # Keep reducing number of data lines until within token limit
    while True:
        truncated = [header] + data_lines
        token_estimate = estimate_tokens("\n".join(truncated))
        if token_estimate <= max_token_limit or len(data_lines) <= 1:
            break
        data_lines = data_lines[:len(data_lines) // 2]

    return "\n".join([header] + data_lines)

# - --- Get Redshift Schema ----
def extract_schema(workgroup_name, database, secret_arn):
    redshift_data = boto3.client('redshift-data')

    logger.info(f"Extracting schema for workgroup: {workgroup_name}, database: {database}")
    """Query to get schema info"""
    query = """
    SELECT table_schema, table_name, column_name, data_type
    FROM information_schema.columns    WHERE table_schema = 'peppa_genbi'
    ORDER BY table_schema, table_name, ordinal_position;
    """
    try:
        print("Extracting schema...")
            # Start the query
        response = redshift_data.execute_statement(
        WorkgroupName=workgroup_name,
        Database=database,
        SecretArn=secret_arn,
        Sql=query
    )
        
        query_id = response['Id']

        # Wait for query to complete
        while True:
            status = redshift_data.describe_statement(Id=query_id)
            if status['Status'] in ['FINISHED', 'FAILED', 'ABORTED']:
                break
            time.sleep(0.5)

        if status['Status'] != 'FINISHED':
            error_message = status.get('Error', 'Unknown error occurred')
            logger.error(f"Query failed: {error_message}")
            raise Exception(f"Redshift query failed: {status['Status']}")

        # Fetch results
        results = redshift_data.get_statement_result(Id=query_id)

        rows = results['Records']

        # Group by table
        schema_map = {}  # (schema.table) -> list of columns
        for row in rows:
            schema_name = row[0]['stringValue']
            table_name = row[1]['stringValue']
            column_name = row[2]['stringValue']
            data_type = row[3]['stringValue']

            full_table_name = f"{schema_name}.{table_name}"
            if full_table_name not in schema_map:
                schema_map[full_table_name] = []
            schema_map[full_table_name].append(f"{column_name} {data_type}")

        # Format to text block
        schema_text_lines = []
        for full_table, columns in schema_map.items():
            formatted = f"{full_table} (\n  " + ",\n  ".join(columns) + "\n)"
            schema_text_lines.append(formatted)

        return "\n\n".join(schema_text_lines)
        
    except Exception as e:
        print(f"Schema extraction failed: {e}")
        logger.error(f"Failed to extract schema: {e}")
        return {}

def schema_to_text(schema):
    """Convert schema dictionary to readable text"""
    return "\n".join(
        f"{table}:\n  - " + "\n  - ".join(cols)
        for table, cols in schema.items()
    )

# ---- Use Nova to Convert Prompt to SQL ----
def clean_sql_query(sql_response):
    """Remove markdown formatting from SQL query"""
    sql_response = sql_response.strip()
    
    # Remove markdown code blocks
    if sql_response.startswith('```sql'):
        sql_response = sql_response[6:]
    elif sql_response.startswith('```'):
        sql_response = sql_response[3:]
    
    if sql_response.endswith('```'):
        sql_response = sql_response[:-3]
    
    # Remove any remaining backticks
    sql_response = sql_response.replace('`', '')
    
    return sql_response.strip()

def get_sql_from_prompt(prompt, schema_text, max_tokens):
    """Generate SQL query from natural language prompt"""
    logger.info(f"Generating SQL for prompt: {prompt}")
    body = {
        "schemaVersion": "messages-v1",
        "system": [
            {
                "text": (
                    f"You are a SQL expert. Generate a valid Redshift SQL query based on the user's question using this database schema:\n\n"
                    f"{schema_text}\n\n"
                    f"Rules:\n"
                    f"- Return ONLY the raw SQL query with no markdown formatting\n"
                    f"- No backticks, no code blocks, no explanations\n"
                    f"- Use proper Redshift syntax\n"
                    f"- Include table aliases for readability\n"
                    f"- Qualify all table names with their schema name \n"
                    f"- Do NOT add a LIMIT clause unless the user explicitly asks for it\n"
                    f"- Query can include more than one datapoint or column if needed\n"
                    f"- End with semicolon"
                )
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": 0.2
        }
    }
    
    try:
        response = bedrock.invoke_model(
            modelId="eu.amazon.nova-pro-v1:0",
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        result = json.loads(response["body"].read())
        raw_sql = result["output"]["message"]["content"][0]["text"]

        # Clean the SQL response
        clean_sql = clean_sql_query(raw_sql)
        
        print(f"Generated SQL: {clean_sql}")  # Debug
        return clean_sql

    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        return None


# ---- Run SQL on Redshift ----
def run_sql(sql_query, workgroup_name, database, secret_arn):
    """Run SQL query on Redshift Serverless and return results"""
    redshift= boto3.client('redshift-data')

    try:
        logger.info(f"Running SQL query: {sql_query}")
        # Start the query
        response = redshift.execute_statement(
            WorkgroupName=workgroup_name,
            Database=database,
            SecretArn=secret_arn,
            Sql=sql_query
        )
        
        statement_id = response['Id']  
        logger.info(f"Started SQL statement with ID: {statement_id}")
        # Step 2: Wait until the statement is finished (polling)
        while True:
            status = redshift.describe_statement(Id=statement_id)
            if status['Status'] in ['FINISHED', 'FAILED', 'ABORTED']:
                break
            time.sleep

        if status['Status'] != 'FINISHED':
            error_message = status.get('Error', 'Unknown error occured')
            logger.error(f"Query failed: {error_message}")
            return {'error': f"Query failed: {error_message}", 'status_code':500, 'data': []}

        logger.info(f"SQL statement finished with status: {status['Status']}")
        # Step 3: Fetch the results
        results = redshift.get_statement_result(Id=statement_id)
        column_names = [col['name'] for col in results['ColumnMetadata']]

        logger.info(f"Retrieved {len(results['Records'])} records with columns: {column_names}")
        # Prepare CSV writing in-memory
        csv_buffer = io.StringIO()
        csv_writer = csv.DictWriter(csv_buffer, fieldnames=column_names)
        csv_writer.writeheader()

        for record in results['Records']:
            row = {}
            for idx, col in enumerate(record):
                val = list(col.values())[0] if col else None
                row[column_names[idx]] = val
            csv_writer.writerow(row)

        logger.info("CSV data written to buffer successfully.")
        # Get CSV string
        csv_text = csv_buffer.getvalue()
        csv_buffer.close()

        # rows = []
        # for record in results['Records']:
        #     row = {}
        #     for idx, col in enumerate(record):
        #         # Redshift Data API returns only one key per column (stringValue, longValue, etc)
        #         val = list(col.values())[0] if col else None
        #         row[column_names[idx]] = val
        #     rows.append(row)
        
        # return {'columns': column_names, 'rows': rows}

        return csv_text

    except Exception as e:
        logger.error(f"Error running SQL query: {e}")
        return {'error': str(e), 'status_code': 500, 'data': []}

def split_response(response: str):
    """Splits model response into text and code parts."""
    lines = response.strip().split("\n")
    text_lines = []
    code_lines = []
    in_code_block = False

    for line in lines:
        if line.strip().startswith("px.") or line.strip().startswith("sns.") or line.strip().startswith("sns.") or line.strip().startswith("df.") or "plt" in line or "sns" in line:
            in_code_block = True
        if in_code_block:
            code_lines.append(line)
        else:
            text_lines.append(line)
    
    return "\n".join(text_lines).strip(), "\n".join(code_lines).strip()

# def split_response(response: str) -> dict:
#     """Splits model response into text and code parts, returns a JSON-style dictionary."""
#     lines = response.strip().split("\n")
#     text_lines = []
#     code_lines = []
#     in_code_block = False

#     for line in lines:
#         stripped = line.strip()

#         # Start collecting code once plot or DataFrame lines appear
#         if (
#             stripped.startswith("plt.") or
#             stripped.startswith("sns.") or
#             stripped.startswith("df.") or
#             "plt" in stripped or
#             "sns" in stripped
#         ):
#             in_code_block = True

#         if in_code_block:
#             code_lines.append(line)
#         else:
#             text_lines.append(line)

#     return {
#         "insight": "\n".join(text_lines).strip(),
#         "code": "\n".join(code_lines).strip()
#     }


# ---- Use Nova to Generate Python Visual Code ----
def get_plot_code(prompt, df_sample, max_tokens, temperature):

    # """Truncate text if token is too large"""
    # max_data_tokens = max_tokens - 1000
    # if estimate_tokens(prompt) + estimate_tokens(df_sample) > max_data_tokens:
    #     logger.warning("Data sample is too large, truncating to fit token limit.")
    #     df_sample = truncate_csv_lines(df_sample, max_data_tokens)


    """Generate Python plotting code using Nova """
    context = (
        "You are a Python expert using pandas and plotly.express. "
        "Given the user prompt {prompt} and a sample of the dataframe {result[:10]}, generate Python code that creates the appropriate chart(s) to visualize the sample data. Note the sample data may be more than the one supplied. "
        "Use only plotly.express. DO NOT include import statements or 'df = ...'. "
        "The dataframe is already available as 'df'. Do not add any import statements, just start with the plotly.express code i.e px.<chart_type>(...). "
        "Return a brief technical summary followed by ONLY the raw Python code. Do NOT include backticks, '```python', or any markdown formatting.\n\n"
        "If the prompt suggests multiple visual insights, generate 2–3 separate plotly.express chart code blocks.\n"
        "Always begin your response with a concise technical summary that provides analytical insight based on the data.\n"
        "Do not include any comments, print statements, or explanatory text—just the summary and the code.\n"
        "The response must be suitable for direct execution in a Python environment—no markdown formatting, no extra text.\n"
    )

    body = {
        "schemaVersion": "messages-v1",
        "system": [{"text": context}],
        "messages": [
            {"role": "user", "content": [{"text": f"Prompt: {prompt}\n\nData Sample:\n{df_sample}"}]}
        ],
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature
        }
    }

    try:
        response = bedrock.invoke_model(
            modelId="eu.amazon.nova-pro-v1:0",
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response["body"].read())
        comb_resp = result["output"]["message"]["content"][0]["text"]
        print(comb_resp)
        logger.info(f"Generated plot code: {comb_resp}")

        return split_response(comb_resp)
        
    except Exception as e:
        logger.error(f"Error generating plot code: {e}")
        return None


def get_data_insight(prompt, df_sample, max_tokens, temperature):
    
    """Truncate text if token is too large"""
    # max_data_tokens = max_tokens - 1000
    # if estimate_tokens(prompt) + estimate_tokens(df_sample) > max_data_tokens:
    #     logger.warning("Data sample is too large, truncating to fit token limit.")
    #     df_sample = truncate_csv_lines(df_sample, max_data_tokens)

    """Generate data insight using Nova"""
    context = (
        "You are a data analyst. Given the user prompt and a sample of the dataframe, generate a concise analytical insight "
        "that summarizes key trends, patterns, or anomalies in the data,  provide recommendations and suggestions where applicable. "
        "Return an extended analysis itemizing each of these trends.\n\n"
    )

    body = {
        "schemaVersion": "messages-v1",
        "system": [{"text": context}],
        "messages": [
            {"role": "user", "content": [{"text": f"Prompt: {prompt}\n\nData Sample:\n{df_sample}"}]}
        ],
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature
        }
    }

    try:
        response = bedrock.invoke_model(
            modelId="eu.amazon.nova-pro-v1:0",
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response["body"].read())

        comb_resp = result["output"]["message"]["content"][0]["text"]
        print(comb_resp)

        return comb_resp

    except Exception as e:
        logger.error(f"Error generating plot code: {e}")
        return None

