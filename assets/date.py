import arrow
from datetime import datetime

def get_current_date():
    current_date = arrow.now().format('YYYY_MM_DD')
    c = datetime.now().time().strftime('%H:%M:%S')
    return current_date + "_" + c