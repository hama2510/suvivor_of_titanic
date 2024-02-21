from datetime import datetime
import uuid

def get_time_string():
    return str(datetime.now().strftime('%s'))

def get_id():
    return str(uuid.uuid4())