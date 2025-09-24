def parse_api_response(status_code, message, data=[]):
    response_headers = {
        "Access-Control-Allow-Headers": "Content-Type",
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    }
    
    
    response_body = {
        "status": "failed" if status_code >= 400 else "success",
        "message": message,
        "data": data
    }
    
    response = {
        "status_code": status_code, 
        "headers": response_headers,
        "data": response_body
    }
    
    return response