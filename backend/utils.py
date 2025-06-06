import os
import uuid
from werkzeug.utils import secure_filename

def allowed_file(filename, allowed_extensions):
    """
    Check if a filename has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def secure_filename_with_uuid(filename):
    """
    Generate a secure filename with a unique UUID prefix to avoid collisions.
    """
    base = secure_filename(filename)
    unique_name = f"{uuid.uuid4().hex}_{base}"
    return unique_name
